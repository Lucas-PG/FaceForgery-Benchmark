"""Independent replication checks and descriptive training-seed sensitivity."""

import numpy as np

from .stats_core import flags, interval, paired_difference


def replication_specs(options, available):
    roles = ("primary_run", "baseline_run", "shuffled_run", "clean_run")
    initial = {k: options[k] for k in roles}
    extra = options.get("replications", [])
    if not isinstance(extra, list):
        raise TypeError("analysis.replications must be a list")
    designs = [initial, *extra]
    for spec in designs:
        if set(spec) != set(roles) or len(set(spec.values())) != len(roles):
            raise ValueError("Each replication must declare four distinct matched runs")
        if not set(spec.values()) <= set(available):
            raise ValueError("Replication references an unscheduled run")
        if len({s.rsplit("__s", 1)[-1] for s in spec.values()}) != 1:
            raise ValueError("Pair replication arms at the same training seed")
    if len({s["primary_run"] for s in designs}) != len(designs):
        raise ValueError("Repeated primary in replication designs")
    return designs


def seed_designs(designs, available):
    result = []
    for spec in designs:
        prefixes = {k: v.rsplit("__s", 1)[0] for k, v in spec.items()}
        for run in available:
            if run.rsplit("__s", 1)[0] != prefixes["primary_run"]:
                continue
            seed = run.rsplit("__s", 1)[-1]
            roles = {k: f"{v}__s{seed}" for k, v in prefixes.items()}
            if set(roles.values()) <= set(available):
                result.append(roles)
    return result


def _pred(stores, cohort, names, panel, policies, role, kind="shared"):
    name = names[role]
    other = stores[name, cohort][0]
    if not np.array_equal(panel.meta.recording_id, other.meta.recording_id):
        raise ValueError("Replication does not share recording identities")
    return flags(other, policies[name][kind])


def outcomes(
    designs, cohorts, stores, policies, ceilings, alpha, gain, clean_margin, tail
):
    rows = []
    for names in designs:
        for cohort in cohorts:
            primary = names["primary_run"]
            results = {"run": primary, "cohort": cohort}
            if any(
                (r, cohort) not in stores
                or not stores[r, cohort][2]
                or not np.isfinite(stores[r, cohort][0].scores).all()
                for r in names.values()
            ):
                rows.append(
                    {
                        **results,
                        "status": "inconclusive",
                        "reason": "unavailable complete scores",
                    }
                )
                continue
            panel = stores[primary, cohort][0]

            p = _pred(stores, cohort, names, panel, policies, "primary_run")
            fpr = p[panel.real].any(axis=1)
            recall = p[~panel.real].all(axis=1)
            results.update(any_fpr=float(fpr.mean()), all_tpr=float(recall.mean()))
            results["fpr_lower"], results["fpr_upper"] = interval(
                int(fpr.sum()), len(fpr), tail
            )
            supported = results["fpr_upper"] <= alpha
            contradicted = results["fpr_lower"] > alpha
            for label, comparator, policy, margin in (
                ("gain", "baseline_run", "shared", gain),
                ("route_gain", "baseline_run", "route", 0.0),
                ("quality_gain", "baseline_run", "quality", 0.0),
                ("correspondence", "shuffled_run", "shared", 0.0),
                ("native", "clean_run", "shared", -clean_margin),
            ):
                d_all = _pred(
                    stores, cohort, names, panel, policies, comparator, policy
                )
                eligible = True
                if label in {"gain", "native"}:
                    reference_fpr = d_all[panel.real].any(axis=1)
                    lo_fp, hi_fp = interval(
                        int(reference_fpr.sum()), len(reference_fpr), tail
                    )
                    results.update(
                        {
                            f"{label}_reference_fpr": float(reference_fpr.mean()),
                            f"{label}_reference_fpr_lower": lo_fp,
                            f"{label}_reference_fpr_upper": hi_fp,
                        }
                    )
                    eligible = hi_fp <= alpha
                    supported &= eligible
                d = d_all[~panel.real]
                a, b = (
                    (p[~panel.real, 0], d[:, 0])
                    if label == "native"
                    else (recall, d.all(axis=1))
                )
                effect, lo, hi = paired_difference(a, b, tail)
                results.update(
                    {label: effect, label + "_lower": lo, label + "_upper": hi}
                )
                supported &= lo > margin
                contradicted |= eligible and hi <= margin
            bound = next(
                (
                    x
                    for x in ceilings
                    if x["run"] == names["baseline_run"] and x["cohort"] == cohort
                ),
                None,
            )
            results["recall_lower"] = interval(int(recall.sum()), len(recall), tail)[0]
            results["ceiling_upper"] = None if bound is None else bound["upper"]
            passed = bool(bound and results["recall_lower"] > bound["upper"])
            results["ceiling_exceeded"] = passed if bound else None
            results["status"] = (
                "supported"
                if supported and passed
                else ("contradicted" if contradicted else "inconclusive")
            )
            rows.append(results)
    return rows
