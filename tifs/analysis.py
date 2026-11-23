"""Sealed analysis of paired video decisions. No model/threshold selection on external outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stats_comparisons import confirm, select_policies
from .stats_core import (
    auc,
    ceiling,
    fit_policies,
    flags,
    interval,
    representatives,
    split_dev,
    transform,
    wide,
)
from .stats_power import confirmation_cohorts, power_audit
from .stats_replication import outcomes, replication_specs, seed_designs
from .stats_transfer import (
    cluster_contrasts,
    extract_subsets,
    mixture_rows,
    recording_outcomes,
    session_rates,
    subgroup_contrasts,
)


def _complete(panel):
    return panel.take(np.isfinite(panel.scores).all(axis=1))


def _panel_inputs(panel, predictions):
    keys = ["recording_id", "route"]
    fields = ["recording_id", "route", "split", "cohort", "family", "cluster_id"]
    panel = panel.copy()
    for name in fields:
        if panel[name].isna().any() or panel[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty {name} in panel")
        panel[name] = panel[name].astype(str)
    if panel.duplicated(keys).any() or not panel.label.isin([0, 1]).all():
        raise ValueError("Panel labels must be binary with one row per recording/route")
    if not set(panel.split) <= {"train", "dev", "cal", "diagnostic", "test"}:
        raise ValueError("Unknown analysis split")
    if (panel.groupby("cluster_id").split.nunique() > 1).any():
        raise ValueError("Lineage crosses a split")
    if (panel.groupby("cluster_id").cohort.nunique() > 1).any():
        raise ValueError("A lineage is assigned to multiple cohorts")
    panel = panel.loc[panel.split != "train"].copy()
    for run, path in predictions.items():
        data = pd.read_csv(path, dtype={k: str for k in keys}, keep_default_na=False)
        raw = data["score"]
        data["score"] = pd.to_numeric(raw, errors="coerce")
        missing = data.score.isna()
        if missing.any() and (
            "failure" not in data
            or data.loc[missing, "failure"].str.strip().eq("").any()
        ):
            raise ValueError(f"{run}: missing score without a failure reason")
        joined = panel.merge(
            data[keys + ["score"]],
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(
                f"{run}: prediction inventory does not match the complete nontraining panel"
            )
        yield run, joined.drop(columns="_merge")


def _metric_rows(run, policy, cohort, panel, decision, tail, family="*"):
    rows = []
    known = np.isfinite(panel.scores)
    real = panel.real
    sub = (
        np.ones(len(real), bool)
        if family == "*"
        else panel.meta.family.to_numpy() == family
    )

    def add(mask, route, metric, values, observed):
        n = int(mask.sum())
        if n:
            events = int(values[mask].sum())
            lo, hi = interval(events, n, tail) if tail is not None else (None, None)
            rows.append(
                {
                    "run": run,
                    "policy": policy,
                    "cohort": cohort,
                    "family": family,
                    "route": route,
                    "metric": metric,
                    "estimate": events / n,
                    "lower": lo,
                    "upper": hi,
                    "n": n,
                    "events": events,
                    "n_fully_scored": int(observed[mask].sum()),
                }
            )

    # Unscored fakes are missed. An unscored genuine could have triggered an alert.
    fake_flag = decision & known
    real_flag = decision | ~known
    add(sub & ~real, "*", "all_tpr", fake_flag.all(axis=1), known.all(axis=1))
    add(sub & real, "*", "any_fpr", real_flag.any(axis=1), known.all(axis=1))
    for i, route in enumerate(panel.routes):
        add(sub & ~real, route, "tpr", fake_flag[:, i], known[:, i])
        add(sub & real, route, "fpr", real_flag[:, i], known[:, i])
    return rows


def _pattern_rows(run, policy, cohort, panel, decision):
    patterns = {}
    for row, d, scores in zip(panel.meta.itertuples(), decision, panel.scores):
        key = (
            int(row.label),
            row.family,
            "".join("?" if np.isnan(s) else str(int(f)) for s, f in zip(scores, d)),
        )
        patterns[key] = patterns.get(key, 0) + 1
    return [
        {
            "run": run,
            "policy": policy,
            "cohort": cohort,
            "routes": "|".join(panel.routes),
            "label": k[0],
            "family": k[1],
            "flags": k[2],
            "n": n,
        }
        for k, n in sorted(patterns.items())
    ]


def _discordance_rows(run, policy, cohort, panel, decision, tail):
    if not np.isfinite(panel.scores).all():
        return []
    out = []
    for metric, values in (
        ("worst_route_minus_all_tpr", decision[~panel.real]),
        ("any_minus_worst_route_fpr", ~decision[panel.real]),
    ):
        # min_c P(correct on c) - P(correct everywhere) counts correct-only-on-some views.
        exclusive = values & ~values.all(axis=1, keepdims=True)
        bounds = [
            interval(int(v), len(values), tail / len(panel.routes))
            for v in exclusive.sum(axis=0)
        ]
        out.append(
            {
                "run": run,
                "policy": policy,
                "cohort": cohort,
                "metric": metric,
                "estimate": float(exclusive.mean(axis=0).min()),
                "lower": min(v[0] for v in bounds),
                "upper": min(v[1] for v in bounds),
                "n": len(values),
            }
        )
    return out


def _score_diagnostics(run, policy_name, cohort, panel, policy, alpha, family="*"):
    # pAUC is normalized by the width alpha: it is average ROC sensitivity over [0,alpha].
    from sklearn.metrics import roc_curve

    result = []
    scores = transform(panel, policy)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    for j, route in enumerate(panel.routes):
        subset = np.isfinite(scores[:, j])
        y, s = (~panel.real)[subset], scores[subset, j]
        if not len(y) or y.all() or not y.any():
            continue
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
        # Interpolate the crossing segment exactly, preserving vertical ties at FPR=0.
        below = fpr <= alpha
        f = np.r_[fpr[below], alpha]
        t = np.r_[tpr[below], np.interp(alpha, fpr, tpr)]
        result.append(
            {
                "run": run,
                "policy": policy_name,
                "cohort": cohort,
                "family": family,
                "route": route,
                "auroc": auc(y, s),
                "partial_auc": float(integrate(t, f) / alpha),
                "n_genuine": int((~y).sum()),
                "n_fake": int(y.sum()),
            }
        )
    return result


def _selection_values(panel, policy):
    if len(_complete(panel).meta) != len(panel.meta):
        return None
    d = flags(panel, policy)
    y = ~panel.real
    return {
        "auc": float(
            np.mean([auc(y, panel.scores[:, i]) for i in range(len(panel.routes))])
        ),
        "all_tpr": float(d[y].all(axis=1).mean()),
        "any_fpr": float(d[~y].any(axis=1).mean()),
        "worst_tpr": float(d[y].mean(axis=0).min()),
        "worst_fpr": float(d[~y].mean(axis=0).max()),
    }


def _run_names(options, runs):
    keys = ["primary_run", "baseline_run", "shuffled_run", "clean_run"]
    names = {key: str(options[key]) for key in keys}
    controls = options.get("control_runs", [])
    if not isinstance(controls, list):
        raise TypeError("analysis.control_runs must be a list of full run IDs")
    selection = options.get(
        "selection_runs",
        list(dict.fromkeys([names["baseline_run"], names["primary_run"], *controls])),
    )
    if not selection or len(set(selection)) != len(selection):
        raise ValueError("selection_runs must be a nonempty list of distinct run IDs")
    if len({r.rsplit("__s", 1)[0] for r in selection}) != len(selection):
        raise ValueError(
            "Selection candidates must not compete over seeds of the same spec"
        )
    if len(set(names.values())) != len(keys):
        raise ValueError("Primary, baseline, clean and shuffled runs must be distinct")
    required = set(names.values()) | set(controls) | set(selection)
    if required - set(runs):
        raise ValueError(f"Missing preregistered runs: {sorted(required - set(runs))}")
    return names, controls, selection


def analyze(cfg: dict, panel: pd.DataFrame, predictions: dict[str, Path], out: Path):
    """Write source-fitted policies, independent sealed estimates and conservative decision gates."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    opt = cfg["analysis"]
    alpha = float(opt.get("alpha", 0.05))
    confidence = float(opt.get("confidence", 0.95))
    source_alpha = float(opt.get("source_alpha", 0.6 * alpha))
    gain, clean_margin = (
        float(opt.get("gain_margin", 0.05)),
        float(opt.get("clean_margin", 0.02)),
    )
    bootstrap = int(opt.get("bootstrap", 500))
    if not (
        0 < source_alpha <= alpha < 0.5
        and 0.5 < confidence < 1
        and 0 <= clean_margin < 1
        and 0 <= gain < 1
        and bootstrap >= 0
    ):
        raise ValueError("Invalid predeclared analysis settings")
    names, controls, selection = _run_names(opt, predictions)
    primary, baseline, shuffled, clean = [
        names[k] for k in ("primary_run", "baseline_run", "shuffled_run", "clean_run")
    ]
    cohorts = opt["cohorts"]
    active = confirmation_cohorts(cfg)
    roles = {c: ("confirmation" if c in active else "transport") for c in cohorts}
    if (
        not cohorts
        or len(set(cohorts)) != len(cohorts)
        or not all(isinstance(c, str) for c in cohorts)
    ):
        raise ValueError("Specify distinct prespecified target cohorts")
    if set(panel.loc[panel.split == "test", "cohort"].astype(str)) != set(cohorts):
        raise ValueError("Mismatch between declared cohorts and external test cohorts")

    designs = replication_specs(opt, predictions)
    seed_pairs = seed_designs(designs, predictions)
    repeat_runs = {r for spec in seed_pairs for r in spec.values()}
    ceiling_runs = {spec["baseline_run"] for spec in designs}
    comparisons = [
        ("matched_baseline", baseline, "shared", gain, "all_tpr"),
        ("channel_calibration", baseline, "route", 0.0, "all_tpr"),
        ("quality_calibration", baseline, "quality", 0.0, "all_tpr"),
        ("correspondence", shuffled, "shared", 0.0, "all_tpr"),
        ("clean_noninferiority", clean, "shared", -clean_margin, "native_tpr"),
    ]
    comparisons += [(f"control_{r}", r, "shared", 0.0, "all_tpr") for r in controls]
    # Allocate error across both sides of each primary rate/difference, ceiling, and
    # adjudication of dev-selected policies. Secondary intervals are marked pointwise.
    tail = (1 - confidence) / (
        2 * len(cohorts) * (len(comparisons) + 18 + 10 * (len(designs) - 1))
    )
    point_tail = (1 - confidence) / 2
    coverage, metrics, patterns, ceiling_rows, dev_rows = [], [], [], [], []
    conditional, all_cases, gaps, ranking, target_cal = [], [], [], [], []
    stores, all_policies, counts = {}, {}, {}
    mixtures, subsets, full_panels, sessions = [], {}, {}, []
    full_comparators = {"augmented": baseline, "shuffled": shuffled, "clean": clean}
    subset_comparators = {"augmented": baseline, "shuffled": shuffled, "clean": clean}
    subset_comparators.update({f"control_{run}": run for run in controls})
    specs = {model["name"]: model for model in cfg.get("models", [])}
    anchor, _, registered_seed = primary.rpartition("__s")
    if anchor in specs:
        for rid in predictions:
            name, _, seed = rid.rpartition("__s")
            spec = specs.get(name, {})
            if (
                seed == registered_seed
                and spec.get("backbone") == specs[anchor]["backbone"]
                and spec.get("objective") in {"per_channel", "no_clean"}
            ):
                subset_comparators[f"ablation_{spec['objective']}"] = rid
    for run, joined in _panel_inputs(panel, predictions):
        src_dev = representatives(wide(joined[joined.split == "dev"]))
        src_cal = representatives(wide(joined[joined.split == "cal"]))
        dev_fit, select = split_dev(src_dev)
        fit, cal = _complete(dev_fit), _complete(src_cal)
        external = {}
        for (split, cohort), rows in joined[
            joined.split.isin(["test", "diagnostic"])
        ].groupby(["split", "cohort"]):
            full = wide(rows)
            cases = representatives(full)
            external[(split, cohort)] = (cases, full)
        for (split, cohort), (cases, full) in external.items():
            for j, route in enumerate(cases.routes):
                for label in (0, 1):
                    mask = cases.meta.label.to_numpy() == label
                    if mask.any():
                        coverage.append(
                            {
                                "run": run,
                                "split": split,
                                "cohort": cohort,
                                "route": route,
                                "label": label,
                                "n": int(mask.sum()),
                                "scored": int(np.isfinite(cases.scores[mask, j]).sum()),
                                "all_recordings": int((full.meta.label == label).sum()),
                            }
                        )
        if fit.meta.label.nunique() < 2 or not cal.real.any():
            all_policies[run] = {
                "unavailable": "insufficient complete source cases for calibration"
            }
            continue
        policies = fit_policies(fit, cal, source_alpha)
        all_policies[run] = policies
        source_ok = len(cal.meta) == len(src_cal.meta) and len(fit.meta) == len(
            dev_fit.meta
        )
        for name, pol in policies.items():
            choice = _selection_values(select, pol)
            if choice is not None and source_ok:
                dev_rows.append(dict(run=run, policy=name, **choice))
        for cohort in cohorts:
            if ("test", cohort) not in external or (
                "diagnostic",
                cohort,
            ) not in external:
                raise ValueError(
                    f"Need external test and diagnostic cohorts for {cohort}"
                )
            test, raw = external[("test", cohort)]
            if test.meta.label.nunique() < 2:
                raise ValueError(
                    f"External cohort {cohort} lacks genuine or fake recordings"
                )
            diag, _ = external[("diagnostic", cohort)]
            if run in {
                primary,
                baseline,
                shuffled,
                clean,
                *controls,
                *selection,
                *repeat_runs,
            }:
                stores[(run, cohort)] = (test, src_cal, source_ok)
            counts[cohort] = {
                "test_records": len(raw.meta),
                "test_lineages": test.meta.cluster_id.nunique(),
                "test_genuine_units": int(test.real.sum()),
                "test_fake_units": int((~test.real).sum()),
                "diagnostic_genuine_units": int(diag.real.sum()),
            }
            if (
                run in ceiling_runs
                and np.isfinite(diag.scores[diag.real]).all()
                and np.isfinite(test.scores).all()
            ):
                bound = ceiling(diag, test, alpha, tail)
                bound["lower_quantiles"] = json.dumps(
                    bound["lower_quantiles"], sort_keys=True
                )
                ceiling_rows.append(dict(run=run, cohort=cohort, **bound))
                if diag.meta.label.nunique() == 2 and np.isfinite(diag.scores).all():
                    try:
                        fit_target, calibrate_target = split_dev(diag)
                    except ValueError:
                        pass  # Diagnostic-only policy requires both labels in two independent folds.
                    else:
                        p = fit_policies(fit_target, calibrate_target, source_alpha)[
                            "route"
                        ]
                        dd = flags(test, p)
                        target_cal.append(
                            {
                                "run": run,
                                "cohort": cohort,
                                "policy": "target_route_supervised",
                                "all_tpr": float(dd[~test.real].all(axis=1).mean()),
                                "any_fpr": float(dd[test.real].any(axis=1).mean()),
                                "policy_json": json.dumps(p, sort_keys=True),
                            }
                        )
            if run in {primary, baseline}:
                sessions.extend(
                    session_rates(joined, run, policies["shared"], cohort, test)
                )
            if bootstrap and source_ok and run in {primary, *full_comparators.values()}:
                full_panels[run, cohort] = recording_outcomes(raw, policies["shared"])
            if source_ok and run in {primary, *subset_comparators.values()}:
                subsets[run, cohort] = extract_subsets(
                    raw,
                    policies["shared"],
                    set(src_cal.routes),
                    opt.get("heldout_families", {}).get(cohort, []),
                )
            family_panels = {
                fam: representatives(
                    raw.take((raw.meta.family == fam).to_numpy() & ~raw.real)
                )
                for fam in sorted(raw.meta.loc[~raw.real, "family"].unique())
            }
            scored_groups = {
                fam: representatives(
                    raw.take(raw.real | (raw.meta.family == fam).to_numpy())
                )
                for fam in family_panels
            }
            # A known route offset or a fitted cutoff shift preserves within-route ranks.
            general_rankings = []
            for pname, policy in policies.items():
                d = flags(test, policy)
                metrics += _metric_rows(run, pname, cohort, test, d, point_tail)
                mixtures += mixture_rows(run, pname, cohort, test, d, point_tail)
                for fam, group in family_panels.items():
                    metrics += _metric_rows(
                        run, pname, cohort, group, flags(group, policy), point_tail, fam
                    )
                patterns += _pattern_rows(run, pname, cohort, test, d)
                gaps += _discordance_rows(run, pname, cohort, test, d, point_tail)
                if pname in {"shared", "quality"}:
                    current = _score_diagnostics(
                        run, pname, cohort, test, policy, alpha
                    )
                    for fam, group in scored_groups.items():
                        current.extend(
                            _score_diagnostics(
                                run, pname, cohort, group, policy, alpha, fam
                            )
                        )
                    ranking.extend(current)
                    if pname == "shared":
                        general_rankings = current
                else:
                    ranking.extend({**row, "policy": pname} for row in general_rankings)
                # Full recording-weighted descriptive estimand and complete-case diagnostic.
                dd = flags(raw, policy)
                for pieces, target in (
                    (raw, all_cases),
                    (_complete(test), conditional),
                ):
                    choice = dd if pieces is raw else flags(pieces, policy)
                    for row in _metric_rows(run, pname, cohort, pieces, choice, None):
                        row.pop("lower")
                        row.pop("upper")
                        target.append(row)
    if not metrics:
        raise ValueError("No evaluable run: examine source scoring and cohort coverage")

    choices = select_policies(dev_rows, selection, alpha)
    contrasts, adjudications, cohort_states, bootrows = confirm(
        cohorts,
        stores,
        all_policies,
        ceiling_rows,
        comparisons,
        primary,
        baseline,
        choices,
        alpha,
        source_alpha,
        tail,
        bootstrap,
        gain,
    )
    rep_checks = outcomes(
        designs[1:],
        cohorts,
        stores,
        all_policies,
        ceiling_rows,
        alpha,
        gain,
        clean_margin,
        tail,
    )
    seed_rows = outcomes(
        seed_pairs,
        cohorts,
        stores,
        all_policies,
        ceiling_rows,
        alpha,
        gain,
        clean_margin,
        point_tail,
    )
    for row in seed_rows:
        row.pop(
            "status", None
        )  # Pointwise seed profiles, never confirmatory inference.
    summary = pd.DataFrame(metrics)
    by_seed = summary[(summary.family == "*") & (summary.route == "*")].copy()
    by_seed["spec"] = by_seed.run.str.replace(r"__s[0-9]+$", "", regex=True)
    seed_stats = (
        by_seed.groupby(["spec", "policy", "cohort", "metric"])
        .estimate.agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    for filename, table, columns in [
        ("metrics", summary, []),
        ("route_mixture", mixtures, []),
        (
            "capture_sessions",
            sessions,
            [
                "cohort",
                "run",
                "route",
                "capture_method",
                "capture_channel",
                "capture_session",
                "capture_settings",
                "label",
                "metric",
                "n",
                "n_scored",
                "events",
                "estimate",
            ],
        ),
        (
            "cluster_sensitivity",
            cluster_contrasts(
                full_panels, primary, full_comparators, cohorts, bootstrap
            ),
            [
                "cohort",
                "comparison",
                "comparator_run",
                "metric",
                "n_recordings",
                "n_clusters",
                "fully_scored",
                "difference",
                "percentile_lower",
                "percentile_upper",
                "replicates",
            ],
        ),
        (
            "subgroup_contrasts",
            subgroup_contrasts(
                subsets, primary, subset_comparators, cohorts, point_tail
            ),
            [
                "cohort",
                "primary_run",
                "comparison",
                "comparator_run",
                "family",
                "route",
                "exposure",
                "label",
                "metric",
                "n",
                "fully_scored",
                "gain",
                "lower",
                "upper",
            ],
        ),
        ("power", power_audit(cfg, panel), []),
        ("seeds", seed_stats, []),
        ("replications", rep_checks, ["run", "cohort", "status"]),
        (
            "seed_contrasts",
            seed_rows,
            ["run", "cohort", "gain", "correspondence", "native"],
        ),
        ("patterns", patterns, []),
        ("coverage", coverage, []),
        (
            "target_calibration",
            target_cal,
            ["run", "cohort", "policy", "all_tpr", "any_fpr", "policy_json"],
        ),
        (
            "gaps",
            gaps,
            ["run", "policy", "cohort", "metric", "estimate", "lower", "upper", "n"],
        ),
        (
            "discrimination",
            ranking,
            [
                "run",
                "policy",
                "cohort",
                "family",
                "route",
                "auroc",
                "partial_auc",
                "n_genuine",
                "n_fake",
            ],
        ),
        ("all_recordings", all_cases, []),
        ("complete_cases", conditional, []),
        (
            "development_selection",
            dev_rows,
            ["run", "policy", "auc", "all_tpr", "any_fpr"],
        ),
        (
            "ceiling",
            ceiling_rows,
            [
                "run",
                "cohort",
                "estimate",
                "upper",
                "relaxed_events",
                "n_fake",
                "n_genuine",
                "order_rank",
                "lower_quantiles",
            ],
        ),
        (
            "contrasts",
            contrasts,
            [
                "cohort",
                "comparison",
                "metric",
                "estimate",
                "lower",
                "upper",
                "margin",
                "status",
            ],
        ),
        (
            "selection",
            adjudications,
            [
                "cohort",
                "comparison",
                "changed_detector",
                "conventional_run",
                "conventional_policy",
                "paired_run",
                "paired_policy",
                "gain",
                "lower",
                "upper",
                "paired_fpr",
                "paired_fpr_lower",
                "paired_fpr_upper",
                "conventional_fpr",
                "conventional_fpr_lower",
                "conventional_fpr_upper",
                "status",
            ],
        ),
        (
            "calibration_bootstrap",
            bootrows,
            [
                "cohort",
                "comparison",
                "replicates",
                "percentile_lower",
                "percentile_upper",
            ],
        ),
    ]:
        frame = pd.DataFrame(table)
        if frame.empty and columns:
            frame = pd.DataFrame(columns=columns)
        if "cohort" in frame and "role" not in frame:
            frame["role"] = frame.cohort.map(roles)
        frame.to_csv(out / f"{filename}.csv", index=False)
    states = {cohort_states[c] for c in active}

    def selection_gate(mode):
        rows = [
            x
            for x in adjudications
            if x["comparison"] == mode and x["cohort"] in active
        ]
        return (
            "supported"
            if len(rows) == len(active)
            and all(x["status"] == "supported" for x in rows)
            else "contradicted"
            if any(x["status"] == "contradicted" for x in rows)
            else "inconclusive"
        )

    decisions = {
        "training_claim": "supported"
        if states == {"supported"}
        else "contradicted"
        if "contradicted" in states
        else "inconclusive",
        "selection_claim": selection_gate("worst_route"),
        "detector_selection_claim": selection_gate("auc_same_shared"),
        "cohorts": cohort_states,
        "cohort_roles": roles,
        "confirmation_cohorts": active,
        "counts": counts,
        "alpha": alpha,
        "source_alpha": source_alpha,
        "confidence": confidence,
        "confirmatory_tail": tail,
        "run_roles": names,
        "comparator_controls": controls,
        "gain_margin": gain,
        "clean_margin": clean_margin,
    }

    def separate_claim(names_needed):
        rows = [
            r
            for r in contrasts
            if r["comparison"] in names_needed and r["cohort"] in active
        ]
        if any(r["status"] == "contradicted" for r in rows):
            return "contradicted"
        if len(rows) == len(active) * len(names_needed) and all(
            r["status"] == "supported" for r in rows
        ):
            return "supported"
        return "inconclusive"

    decisions["matched_training_claim"] = separate_claim(
        {r[0] for r in comparisons}
        | {
            "false_alarm_budget",
            "eligibility_matched_baseline",
            "eligibility_clean_noninferiority",
        }
    )
    decisions["beyond_recalibration_claim"] = separate_claim(
        {"false_alarm_budget", "recalibration_ceiling"}
    )
    seed_consistent = bool(seed_rows) and all(
        row.get("gain", -1) > 0
        and row.get("correspondence", -1) > 0
        and row.get("native", -1) >= -clean_margin
        and row.get("any_fpr", 1) <= alpha
        and row.get("gain_reference_fpr", 1) <= alpha
        and row.get("native_reference_fpr", 1) <= alpha
        for row in seed_rows
        if row["cohort"] in active
    )
    decisions["all_seed_directions_consistent"] = seed_consistent
    rep_states = {r["status"] for r in rep_checks if r["cohort"] in active}
    decisions["replicated_training_claim"] = (
        "not_preregistered"
        if not rep_checks
        else "supported"
        if decisions["training_claim"] == "supported"
        and rep_states == {"supported"}
        and seed_consistent
        else "contradicted"
        if decisions["training_claim"] == "contradicted" or "contradicted" in rep_states
        else "inconclusive"
    )
    allow_incomplete = cfg.get("data", {}).get("allow_incomplete", False)
    if cfg.get("mode") == "exploratory" or allow_incomplete:
        for field in (
            "training_claim",
            "matched_training_claim",
            "beyond_recalibration_claim",
            "replicated_training_claim",
            "selection_claim",
            "detector_selection_claim",
        ):
            decisions[field] = "exploratory"
    if allow_incomplete:
        decisions["attrition_note"] = (
            "Only completed acquisition panels were analyzed; inspect panel/attrition.csv."
        )
    for name, obj in (("decisions", decisions), ("policies", all_policies)):
        (out / f"{name}.json").write_text(
            json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=int)
            + "\n"
        )
    return decisions
