"""Create camera-ready tables, figures and a claim-by-claim summary from a sealed analysis."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .io import atomic_text, sha, write_json


def latex(value):
    escapes = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(ch, ch) for ch in str(value))


def _tex(path, frame):
    # These are table fragments for single/double-column LaTeX. Full-precision
    # machine-readable data are emitted separately, without formatted percentages.
    if frame.empty:
        atomic_text(path, r"\emph{No eligible observations for this table.}" + "\n")
        return
    lines = [
        r"\begin{tabular}{" + "l" * len(frame.columns) + "}",
        r"\toprule",
        " & ".join(map(latex, frame.columns)) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            " & ".join(latex(v) if pd.notna(v) else "---" for v in row) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    atomic_text(path, "\n".join(lines))


def _interval(r):
    return f"{r.estimate:.1%} [{r.lower:.1%}, {r.upper:.1%}]"


def _case_table(frame, index):
    rows = []
    for values, group in frame.groupby(index, sort=True):
        vals = (
            (values,) if len(index) == 1 and not isinstance(values, tuple) else values
        )
        row = dict(zip(index, vals))
        for r in group.itertuples():
            field = {
                "all_tpr": "All-copy sensitivity",
                "any_fpr": "Any-copy false alerts",
                "tpr": "Sensitivity",
                "fpr": "False alerts",
            }[r.metric]
            row[field] = f"{r.events}/{r.n}  {_interval(r)}"
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tables(cfg, data, dest):
    # Keep source analysis CSVs byte-for-byte so denominators, precision, abstentions,
    # subgroup losses and exploratory comparisons remain available beside the main tables.
    files = sorted(data.glob("*.csv"))
    for file in files:
        shutil.copyfile(file, dest / file.name)
    for name in ("policies.json", "decisions.json"):
        shutil.copyfile(data / name, dest / name)
    roles = cfg["analysis"]
    focus = {
        roles[k] for k in ("primary_run", "baseline_run", "shuffled_run", "clean_run")
    }
    focus.update(roles.get("control_runs", []))
    focus.update(roles.get("selection_runs", []))
    for design in roles.get("replications", []):
        focus.update(design.values())
    metrics = pd.read_csv(data / "metrics.csv")
    original = metrics[(metrics.run.isin(focus)) & metrics.policy.eq("shared")]
    units = original[original.family == "*"]
    macro = units[units.route == "*"]
    for name, frame, index in (
        ("recording_errors", macro, ["cohort", "run"]),
        ("route_performance", units[units.route != "*"], ["cohort", "route", "run"]),
        (
            "generator_sensitivity",
            original[
                (original.family != "*")
                & original.metric.eq("all_tpr")
                & original.run.isin([roles["primary_run"], roles["baseline_run"]])
            ],
            ["cohort", "family", "run"],
        ),
        (
            "calibration_comparison",
            metrics[
                (metrics.family == "*")
                & (metrics.route == "*")
                & (metrics.run == roles["baseline_run"])
            ],
            ["cohort", "policy"],
        ),
    ):
        cols = list(
            dict.fromkeys(
                [
                    *index,
                    "metric",
                    "events",
                    "n",
                    "estimate",
                    "lower",
                    "upper",
                    "n_fully_scored",
                ]
            )
        )
        frame[cols].to_csv(dest / (name + ".csv"), index=False)
        _tex(dest / (name + ".tex"), _case_table(frame, index))
    # Different bounds in the ceiling gate must not appear as an ordinary
    # confidence interval on one quantity.
    effects = pd.read_csv(data / "contrasts.csv")
    differences = effects[
        ~effects.comparison.isin(["false_alarm_budget", "recalibration_ceiling"])
        & ~effects.comparison.str.startswith("eligibility_")
    ]
    rows = [
        {
            "Cohort": r.cohort,
            "Comparison": r.comparison,
            "Difference, pp": f"{100 * r.estimate:.1f} [{100 * r.lower:.1f}, {100 * r.upper:.1f}]",
            "Required, pp": f"{100 * r.margin:.1f}",
            "Evidence": r.status,
        }
        for r in differences.itertuples()
    ]
    _tex(
        dest / "contrasts.tex",
        pd.DataFrame(
            rows,
            columns=[
                "Cohort",
                "Comparison",
                "Difference, pp",
                "Required, pp",
                "Evidence",
            ],
        ),
    )
    bounds = effects[effects.comparison == "recalibration_ceiling"]
    table = [
        {
            "Cohort": r.cohort,
            "Learned recall, lower": f"{r.lower:.1%}",
            "Frozen ceiling, upper": f"{r.upper:.1%}",
            "Evidence": r.status,
        }
        for r in bounds.itertuples()
    ]
    _tex(
        dest / "ceiling.tex",
        pd.DataFrame(
            table,
            columns=[
                "Cohort",
                "Learned recall, lower",
                "Frozen ceiling, upper",
                "Evidence",
            ],
        ),
    )
    selection = pd.read_csv(data / "selection.csv")
    table = [
        {
            "Cohort": r.cohort,
            "Source rule": r.comparison,
            "Conventional": r.conventional_run,
            "Paired": r.paired_run,
            "Gain, pp": f"{100 * r.gain:.1f}",
            "Evidence": r.status,
        }
        for r in selection.itertuples()
    ]
    _tex(
        dest / "selection.tex",
        pd.DataFrame(
            table,
            columns=[
                "Cohort",
                "Source rule",
                "Conventional",
                "Paired",
                "Gain, pp",
                "Evidence",
            ],
        ),
    )
    return metrics, macro, effects


def _figure(joint, cohort, budget, path):
    view = joint[joint.cohort == cohort]
    if view.empty:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(8, max(3, 0.38 * view.run.nunique() + 1.2)), layout="constrained"
    )
    for ax, metric, label in zip(
        axes,
        ("all_tpr", "any_fpr"),
        ("All-copy sensitivity", "Any-copy genuine false alerts"),
    ):
        v = view[view.metric == metric].sort_values("run")
        y = np.arange(len(v))
        ax.errorbar(
            v.estimate,
            y,
            xerr=np.maximum(0, np.array([v.estimate - v.lower, v.upper - v.estimate])),
            fmt="o",
            capsize=2,
            color="#164968",
            markersize=4,
        )
        ax.set_yticks(y, v.run if ax is axes[0] else [])
        lim = (
            1.0
            if metric == "all_tpr"
            else min(1.0, max(1.5 * budget, 1.12 * float(v.upper.max()), 0.1))
        )
        ax.set_xlim(-0.005, lim + 0.005)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.2)
        if metric == "any_fpr":
            ax.axvline(
                budget, color="#a33326", ls="--", lw=1, label=f"Budget {budget:.0%}"
            )
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{cohort}  |  Shared source-calibrated cutoff")
    fig.savefig(
        path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def _recalibration_figure(metrics, effects, cohort, roles, path):
    m = metrics[
        (metrics.cohort == cohort)
        & (metrics.metric == "all_tpr")
        & (metrics.family == "*")
    ]
    baseline = m[m.run == roles["baseline_run"]].copy().sort_values("policy")
    baseline["label"] = "Frozen / " + baseline.policy
    train = m[(m.run == roles["primary_run"]) & (m.policy == "shared")].copy()
    train["label"] = "Trained / shared"
    frame = pd.concat([baseline, train])
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6.3, 3.4), layout="constrained")
    y = np.arange(len(frame))
    ax.errorbar(
        frame.estimate,
        y,
        xerr=np.maximum(
            0, np.array([frame.estimate - frame.lower, frame.upper - frame.estimate])
        ),
        fmt="o",
        capsize=2,
        color="#164968",
        label="Estimate and pointwise interval",
    )
    row = effects[
        (effects.cohort == cohort) & (effects.comparison == "recalibration_ceiling")
    ]
    if len(row):
        ax.axvline(
            row.iloc[0].upper,
            color="#a33326",
            ls="--",
            label="Frozen monotone ceiling, upper limit",
        )
        ax.plot(
            row.iloc[0].lower,
            len(frame) - 1,
            "D",
            markersize=4,
            color="#242424",
            label="Trained recall, simultaneous lower limit",
        )
    ax.set_yticks(y, frame.label)
    ax.set_xlim(-0.015, 1.015)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Forged recordings flagged in every registered copy")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(str(cohort))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.20))
    fig.savefig(
        path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def _crop_evidence(out, stage):
    source = out / "crop"
    receipt = json.loads((source / "complete.json").read_text())
    required = {"result": "comparison.csv", "effects": "paired_effects.csv"}
    if receipt["analysis"] != sha(out / "analysis/complete.json") or any(
        sha(source / name) != receipt[key] for key, name in required.items()
    ):
        raise ValueError("Crop evidence does not match its frozen analysis receipt")
    for name in required.values():
        shutil.copyfile(source / name, stage / ("crop_" + name))
    effects = pd.read_csv(source / "paired_effects.csv")
    subset = effects[effects.comparison == "tracked_minus_end_to_end:primary_gain"]
    pretty = []
    notes = []
    for r in subset.itertuples():
        direction = (
            "higher" if r.lower > 0 else "lower" if r.upper < 0 else "unresolved"
        )
        # Percentile intervals on identical outcomes can collapse at zero with a
        # tiny fixed-track sample. Never interpret those as proof of equivalence.
        notes.append(
            f"- {r.cohort}, {r.metric}: switching to fixed tracks changed the "
            f"primary-minus-baseline difference by {100 * r.estimate:+.1f} pp "
            f"[{100 * r.lower:+.1f}, {100 * r.upper:+.1f}] on {r.n} independent units; "
            f"direction {direction} (diagnostic interval)."
        )
        pretty.append(
            {
                "Cohort": r.cohort,
                "Metric": r.metric,
                "Cases": r.n,
                "Change in gain, pp": f"{100 * r.estimate:+.1f} [{100 * r.lower:+.1f}, {100 * r.upper:+.1f}]",
            }
        )
    _tex(stage / "crop_paired_effects.tex", pd.DataFrame(pretty))
    return [
        "",
        "## Fixed-track subpopulation",
        "",
        *notes,
        "These paired crop comparisons concern only registered tracked clips, unchanged weights and frozen cutoffs.",
        "",
    ]


def _summary(cfg, data, out, decisions, joint, effects):
    opt = cfg["analysis"]
    fixture = cfg.get("mode") == "fixture"
    intro = ["# Recording-level evidence", ""]
    if fixture:
        intro += [
            "**Synthetic engineering fixture. All rates below test the pipeline and have no empirical bearing on deepfakes.**",
            "",
        ]
    for label, key in (
        ("Primary checkpoint and seed, all registered gates", "training_claim"),
        ("Matched training and calibration comparators", "matched_training_claim"),
        (
            "Learned sensitivity beyond frozen route recalibration",
            "beyond_recalibration_claim",
        ),
        ("Cross-backbone and seed replication", "replicated_training_claim"),
        ("Choice of detector and operating policy", "selection_claim"),
        (
            "Detector choice at the identical shared-calibration rule",
            "detector_selection_claim",
        ),
    ):
        intro.append(f"- {label}: **{decisions[key]}**.")
    intro += [
        "",
        (
            f"The registered any-copy genuine false-alert budget is {decisions['alpha']:.1%}. "
            f"The material sensitivity gain is {decisions['gain_margin']:.1%} and the tolerated native loss is "
            f"{decisions['clean_margin']:.1%} absolute. Gates use simultaneous limits. The intervals on ordinary rates are pointwise."
        ),
        "",
    ]
    if "attrition_note" in decisions:
        intro += [
            decisions["attrition_note"],
            "The following cohort gate details are conditional on completed acquisition.",
            "",
        ]
    selection = pd.read_csv(data / "selection.csv")
    power = pd.read_csv(data / "power.csv")
    for cohort, status in decisions["cohorts"].items():
        counts = decisions["counts"][cohort]
        intro += [
            f"## {cohort}",
            "",
            (
                f"Registered role: **{decisions['cohort_roles'][cohort]}**. Within-cohort result: **{status}**. Independent units by class: "
                f"{counts['test_genuine_units']} genuine and {counts['test_fake_units']} forged."
            ),
            "",
        ]
        for r in joint[
            (joint.cohort == cohort) & (joint.run == opt["primary_run"])
        ].itertuples():
            label = (
                "Forged recordings flagged throughout"
                if r.metric == "all_tpr"
                else "Genuine recordings potentially flagged in any rendition"
            )
            intro.append(f"- {label}: {r.events}/{r.n}, {_interval(r)}.")
        c = effects[effects.cohort == cohort]
        for r in c.itertuples():
            if r.comparison == "recalibration_ceiling":
                intro.append(
                    f"- Threshold explanation: **{r.status}** for exceeding the fixed-score bound. "
                    f"Trained lower sensitivity limit {r.lower:.1%}, frozen monotone upper limit {r.upper:.1%}."
                )
            elif r.comparison.startswith("eligibility_"):
                intro.append(
                    f"- {r.comparison}: **{r.status}**. Reference any-copy false-alert upper limit {r.upper:.1%} against {r.margin:.1%}."
                )
            elif r.comparison == "false_alarm_budget":
                intro.append(
                    f"- Any-copy budget: **{r.status}**. Upper false-alert limit {r.upper:.1%} against {r.margin:.1%}."
                )
            else:
                intro.append(
                    f"- {r.comparison}: **{r.status}**. Difference {r.estimate * 100:+.1f} pp "
                    f"[{100 * r.lower:+.1f}, {100 * r.upper:+.1f}], required above {100 * r.margin:+.1f} pp."
                )
        if c.empty:
            intro.append(
                "- No complete, eligible primary comparison. Inspect coverage and scoring failures."
            )
        for r in selection[
            (selection.cohort == cohort)
            & selection.comparison.isin(["worst_route", "auc_same_shared"])
        ].itertuples():
            intro.append(
                f"- {r.comparison}: {r.conventional_run}/{r.conventional_policy} versus {r.paired_run}/{r.paired_policy}. "
                f"Independent paired-minus-conventional all-copy sensitivity {100 * r.gain:+.1f} pp "
                f"[{100 * r.lower:+.1f}, {100 * r.upper:+.1f}]."
            )
        for r in power[power.cohort == cohort].itertuples():
            intro.append(
                f"- At the registered planning rates, projected power is {r.fpr_power:.1%} for one false-alert cap, "
                f"{r.gain_power:.1%} for the material gain, {r.clean_power:.1%} for native noninferiority and "
                f"{r.reference_ceiling_power:.1%} for the reference ceiling scenario. These are separate design projections."
            )
        intro.append("")
    audit = json.loads((out / "panel/panel_audit.json").read_text())
    coverage = pd.read_csv(data / "coverage.csv")
    intro += [
        "## Reading the evidence",
        "",
        (
            f"Enrolled {audit['enrolled_recordings']} recordings; {audit['included_recordings']} had complete acquisition. "
            f"The representative-unit coverage table contains {int((coverage.n - coverage.scored).sum())} missing model scores "
            "across model, split, route and class entries."
        ),
        (
            "Unscored forged cases count as misses and unscored genuine cases as possible false alarms in enrolled-case summaries. "
            "Inspect coverage.csv and complete_cases.csv for the conditional comparison."
        ),
        "",
        (
            "A result below the recalibration upper limit leaves the possibility of a better threshold policy open. "
            "That ceiling covers route-specific monotone transformations of the frozen score; quality-based models that reorder "
            "cases have a separate head-to-head comparison. An inconclusive difference does not establish equivalence."
        ),
        (
            "See selection.csv for equal-information source selection, replications.csv and seed_contrasts.csv for reproducibility, "
            "subgroup_contrasts.csv for matched ablations, held-out families and routes, route_mixture.csv for an equal-route operating policy, and power.csv for design assumptions. All-copy and any-copy outcomes refer to the stated finite route panel."
        ),
        "LaTeX fragments require booktabs. Percentages are rounded only for display; CSVs retain full precision.",
        "",
    ]
    return "\n".join(intro)


def report(cfg, out):
    """Write verified report files under a temporary name, replacing the artifact index last."""
    out = Path(out)
    data, dest = out / "analysis", out / "paper"
    dest.mkdir(parents=True, exist_ok=True)
    decisions = json.loads((data / "decisions.json").read_text())
    mpl = {"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
    with TemporaryDirectory(prefix=".paper-", dir=out) as tmp, plt.rc_context(mpl):
        stage = Path(tmp)
        metrics, joint, effects = _write_tables(cfg, data, stage)
        crop_notes = _crop_evidence(out, stage) if cfg.get("crop_diagnostic") else []
        for i, cohort in enumerate(cfg["analysis"]["cohorts"], 1):
            _figure(
                joint,
                cohort,
                decisions["alpha"],
                stage / f"cohort_{i:02d}_decisions.pdf",
            )
            _recalibration_figure(
                metrics,
                effects,
                cohort,
                cfg["analysis"],
                stage / f"cohort_{i:02d}_calibration.pdf",
            )
        atomic_text(
            stage / "SUMMARY.md",
            _summary(cfg, data, out, decisions, joint, effects) + "\n".join(crop_notes),
        )
        files = {f.name: sha(f) for f in sorted(stage.iterdir())}
        old = dest / "ARTIFACTS.json"
        prior = json.loads(old.read_text()).get("files", {}) if old.is_file() else {}
        for name in files:
            (stage / name).replace(dest / name)
        for name in set(prior) - set(files):
            # Only remove this generator's own stale artifacts and never traverse paths.
            if (
                Path(name).name == name
                and (dest / name).is_file()
                and sha(dest / name) == prior[name]
            ):
                (dest / name).unlink()
        write_json(
            old,
            {
                "protocol": sha(out / "protocol.json"),
                "analysis": sha(data / "complete.json"),
                "panel_audit": sha(out / "panel/panel_audit.json"),
                "crop": sha(out / "crop/complete.json") if crop_notes else None,
                "files": files,
            },
        )
