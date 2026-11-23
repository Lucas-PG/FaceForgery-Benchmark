"""Blinded model selection and independent confirmatory decision contrasts."""

import numpy as np

from .stats_core import cutoff_scores, flags, interval, paired_difference, transform


def add(rows, cohort, name, metric, x, l, u, margin, greater=True):
    supported = l > margin if greater else u <= margin
    contradicted = u <= margin if greater else l > margin
    status = (
        "supported"
        if supported
        else ("contradicted" if contradicted else "inconclusive")
    )
    rows.append(
        {
            "cohort": cohort,
            "comparison": name,
            "metric": metric,
            "estimate": x,
            "lower": l,
            "upper": u,
            "margin": margin,
            "status": status,
        }
    )


def get(stores, cohort, run):
    st = stores.get((run, cohort))
    if st and st[2] and np.isfinite(st[0].scores).all():
        return st[:2]
    return None


def selected_flags(sel, stores, cohort, all_policies):
    m = get(stores, cohort, sel["run"])[0]
    return m, flags(m, all_policies[sel["run"]][sel["policy"]])


def _sensitivity_bootstrap(left, right, bootstrap, source_alpha, rng):
    """Paired cluster resampling of source calibration and sealed test conditional on selected models.

    Each case matrix here contains one representative per cluster and class. The same
    sampled IDs are used for both competitors; development-fitted transforms remain fixed.
    """
    left_cal, left_test, lp = left
    right_cal, right_test, rp = right
    if not (
        np.array_equal(left_cal.meta.recording_id, right_cal.meta.recording_id)
        and np.array_equal(left_test.meta.recording_id, right_test.meta.recording_id)
    ):
        raise ValueError("Mismatched bootstrap populations")
    if not (np.isfinite(left_cal.scores).all() and np.isfinite(right_cal.scores).all()):
        return None
    genuine = left_cal.real
    fakes = ~left_test.real
    cal_values = [
        transform(z, p)[genuine] for z, p in ((left_cal, lp), (right_cal, rp))
    ]
    test_values = [
        transform(z, p)[fakes] for z, p in ((left_test, lp), (right_test, rp))
    ]
    delta = np.empty(bootstrap)
    for b in range(bootstrap):
        cal_idx = rng.integers(genuine.sum(), size=genuine.sum())
        test_idx = rng.integers(fakes.sum(), size=fakes.sum())
        rates = []
        for base, test, p in zip(cal_values, test_values, (lp, rp)):
            cutoff = cutoff_scores(
                base[cal_idx], p["kind"], source_alpha, left_cal.routes
            )
            if isinstance(cutoff, dict):
                cutoff = [cutoff.get(r, max(cutoff.values())) for r in left_test.routes]
            rates.append(np.mean(np.all(test[test_idx] > cutoff, axis=1)))
        delta[b] = rates[0] - rates[1]
    return tuple(map(float, np.quantile(delta, [0.025, 0.975])))


def select_policies(dev_rows, selection, alpha):
    eligible = [r for r in dev_rows if r["run"] in selection]
    # Both selectors observe the same development panel, calibrated candidates and routes.
    # Only their eligibility criterion and sensitivity objective differ.
    candidates = [r for r in eligible if r["policy"] in {"shared", "route", "quality"}]
    conventional_options = [r for r in candidates if r["worst_fpr"] <= alpha]
    paired_options = [r for r in candidates if r["any_fpr"] <= alpha]
    conventional = max(
        conventional_options,
        key=lambda r: (r["worst_tpr"], -r["worst_fpr"], r["run"], r["policy"]),
        default=None,
    )
    paired = max(
        paired_options,
        key=lambda r: (r["all_tpr"], -r["any_fpr"], r["run"], r["policy"]),
        default=None,
    )
    shared = [r for r in candidates if r["policy"] == "shared"]
    auc_winner = max(shared, key=lambda r: (r["auc"], r["run"]), default=None)
    paired_shared = max(
        (r for r in shared if r["any_fpr"] <= alpha),
        key=lambda r: (r["all_tpr"], -r["any_fpr"], r["run"]),
        default=None,
    )
    auc_calibrated = max(
        (r for r in paired_options if auc_winner and r["run"] == auc_winner["run"]),
        key=lambda r: (r["all_tpr"], -r["any_fpr"], r["policy"]),
        default=None,
    )
    auc_native = next(
        (
            r
            for r in eligible
            if auc_winner and r["run"] == auc_winner["run"] and r["policy"] == "native"
        ),
        None,
    )
    return (
        ("worst_route", conventional, paired),
        ("auc_same_shared", auc_winner, paired_shared),
        ("auc_best_source_calibrated", auc_calibrated, paired),
        ("auc_native_diagnostic", auc_native, paired),
    )


def confirm(
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
):
    rng = np.random.default_rng(20261123)
    contrasts, adjudications, cohort_states, bootrows = [], [], {}, []
    for cohort in cohorts:
        rows = []

        lead = get(stores, cohort, primary)
        if lead:
            test, lead_cal = lead
            d = flags(test, all_policies[primary]["shared"])
            any_fp = d[test.real].any(axis=1)
            fake = ~test.real
            persistent = d[fake].all(axis=1)
            add(
                rows,
                cohort,
                "false_alarm_budget",
                "any_fpr",
                float(any_fp.mean()),
                *interval(int(any_fp.sum()), len(any_fp), tail),
                alpha,
                False,
            )
            for name, other, pname, margin, metric in comparisons:
                if not get(stores, cohort, other):
                    continue
                others, other_cal = get(stores, cohort, other)
                if not np.array_equal(test.meta.recording_id, others.meta.recording_id):
                    raise ValueError(
                        "Competitors were evaluated on different recording representatives"
                    )
                other_decision = flags(others, all_policies[other][pname])
                if name in {"matched_baseline", "clean_noninferiority"}:
                    fp = other_decision[others.real].any(axis=1)
                    low, high = interval(int(fp.sum()), len(fp), tail)
                    rows.append(
                        {
                            "cohort": cohort,
                            "comparison": f"eligibility_{name}",
                            "metric": "reference_any_fpr",
                            "estimate": float(fp.mean()),
                            "lower": low,
                            "upper": high,
                            "margin": alpha,
                            "status": "supported"
                            if high <= alpha
                            else ("ineligible" if low > alpha else "inconclusive"),
                        }
                    )
                od = other_decision[fake]
                if metric == "all_tpr":
                    a, b = persistent, od.all(axis=1)
                else:
                    a, b = d[fake, 0], od[:, 0]
                add(rows, cohort, name, metric, *paired_difference(a, b, tail), margin)
                if name == "matched_baseline" and bootstrap:
                    bounds = _sensitivity_bootstrap(
                        (lead_cal, test, all_policies[primary]["shared"]),
                        (other_cal, others, all_policies[other][pname]),
                        bootstrap,
                        source_alpha,
                        rng,
                    )
                    if bounds:
                        bootrows.append(
                            {
                                "cohort": cohort,
                                "comparison": name,
                                "replicates": bootstrap,
                                "percentile_lower": bounds[0],
                                "percentile_upper": bounds[1],
                            }
                        )
            bound = next(
                (
                    v
                    for v in ceiling_rows
                    if v["run"] == baseline and v["cohort"] == cohort
                ),
                None,
            )
            if bound is not None:
                ll, _ = interval(int(persistent.sum()), len(persistent), tail)
                rows.append(
                    {
                        "cohort": cohort,
                        "comparison": "recalibration_ceiling",
                        "metric": "all_tpr",
                        "estimate": float(persistent.mean()),
                        "lower": ll,
                        "upper": bound["upper"],
                        "margin": None,
                        "status": "supported"
                        if ll > bound["upper"]
                        else "inconclusive",
                    }
                )
        invalid = {
            r["comparison"].removeprefix("eligibility_")
            for r in rows
            if r["comparison"].startswith("eligibility_") and r["status"] != "supported"
        }
        for row in rows:
            if row["comparison"] in invalid:
                row["status"] = "reference_eligibility_unverified"
        contrasts.extend(rows)
        expected = len(comparisons) + 4
        cohort_states[cohort] = (
            "supported"
            if len(rows) == expected and all(r["status"] == "supported" for r in rows)
            else "contradicted"
            if any(r["status"] == "contradicted" for r in rows)
            else "inconclusive"
        )

        for mode, conventional_choice, paired_choice in choices:
            if not (
                conventional_choice
                and paired_choice
                and get(stores, cohort, conventional_choice["run"])
                and get(stores, cohort, paired_choice["run"])
            ):
                continue

            p, df = selected_flags(paired_choice, stores, cohort, all_policies)
            _, dc = selected_flags(conventional_choice, stores, cohort, all_policies)
            effect, lower, upper = paired_difference(
                df[~p.real].all(axis=1), dc[~p.real].all(axis=1), tail
            )
            fpa, fpc = df[p.real].any(axis=1), dc[p.real].any(axis=1)
            lo_p, up_p = interval(int(fpa.sum()), len(fpa), tail)
            lo_c, up_c = interval(int(fpc.sum()), len(fpc), tail)
            same_rule = all(
                conventional_choice[k] == paired_choice[k] for k in ("run", "policy")
            )
            ruled_out = same_rule or lo_p > alpha or (upper <= gain and up_c <= alpha)
            adjudications.append(
                {
                    "cohort": cohort,
                    "comparison": mode,
                    "changed_detector": conventional_choice["run"]
                    != paired_choice["run"],
                    "conventional_run": conventional_choice["run"],
                    "conventional_policy": conventional_choice["policy"],
                    "paired_run": paired_choice["run"],
                    "paired_policy": paired_choice["policy"],
                    "gain": effect,
                    "lower": lower,
                    "upper": upper,
                    "paired_fpr": float(fpa.mean()),
                    "paired_fpr_lower": lo_p,
                    "paired_fpr_upper": up_p,
                    "conventional_fpr": float(fpc.mean()),
                    "conventional_fpr_lower": lo_c,
                    "conventional_fpr_upper": up_c,
                    "status": "supported"
                    if not same_rule
                    and up_p <= alpha
                    and (lower > gain or lo_c > alpha)
                    else ("contradicted" if ruled_out else "inconclusive"),
                }
            )

    return contrasts, adjudications, cohort_states, bootrows
