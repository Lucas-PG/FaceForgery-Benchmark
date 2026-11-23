"""Prespecified subset and mechanism checks, conditional on registered checkpoints."""

import numpy as np

from .stats_core import flags, interval, paired_difference, representatives


def mixture_rows(run, policy, cohort, panel, decision, tail):
    """Equal route probability per recording; simultaneous marginal bounds allow dependence."""
    known = np.isfinite(panel.scores)
    rows = []
    for label, values, metric in (
        (0, decision | ~known, "mean_fpr"),
        (1, decision & known, "mean_tpr"),
    ):
        sample = values[(~panel.real) == label]
        n = len(sample)
        if not n:
            continue
        bounds = [interval(int(k), n, tail / len(panel.routes)) for k in sample.sum(0)]
        rows.append(
            {
                "run": run,
                "policy": policy,
                "cohort": cohort,
                "metric": metric,
                "estimate": float(sample.mean()),
                "lower": float(np.mean([lo for lo, _ in bounds])),
                "upper": float(np.mean([hi for _, hi in bounds])),
                "n_recordings": n,
                "n_routes": len(panel.routes),
            }
        )
    return rows


def extract_subsets(raw, policy, source_routes, families):
    """Use a fresh, score-independent representative within each declared fake family."""
    tests = [("*", representatives(raw))]
    for family in families:
        mask = (
            raw.meta.family.str.casefold() == family.casefold()
        ).to_numpy() & ~raw.real
        if mask.any():
            tests.append((family, representatives(raw.take(mask))))
    subgroups = {}
    for family, pop in tests:
        d = flags(pop, policy)
        for label in (0, 1):
            rows = (~pop.real) == label
            if not rows.any():
                continue
            locations = {"*": None, **{r: j for j, r in enumerate(pop.routes)}}
            for route, j in locations.items():
                values = d[rows]
                available = np.isfinite(pop.scores[rows])
                if j is None:
                    event = values.all(1) if label else values.any(1)
                    full = available.all(1)
                    setting = "joint"
                else:
                    event, full = values[:, j], available[:, j]
                    setting = (
                        "known_route" if route in source_routes else "held_out_route"
                    )
                subgroups[(family, route, label)] = (
                    pop.meta.loc[rows, "recording_id"].to_numpy(),
                    event,
                    full,
                    setting,
                )
    return subgroups


def subgroup_contrasts(subsets, primary, comparators, cohorts, tail):
    out = []
    for cohort in cohorts:
        left = subsets.get((primary, cohort), {})
        for name, run in comparators.items():
            right = subsets.get((run, cohort), {})
            for (family, route, label), (ids, a, complete_a, setting) in left.items():
                if (family, route, label) not in right:
                    continue
                ids_b, b, complete_b, _ = right[family, route, label]
                if not np.array_equal(ids, ids_b):
                    raise ValueError(
                        "Different recording representatives in matched subgroup"
                    )
                scored = complete_a & complete_b
                delta, lo, hi = (np.nan,) * 3
                if scored.all():
                    delta, lo, hi = paired_difference(a, b, tail)
                out.append(
                    {
                        "cohort": cohort,
                        "primary_run": primary,
                        "comparison": name,
                        "comparator_run": run,
                        "family": family,
                        "route": route,
                        "exposure": setting,
                        "label": label,
                        "metric": ("all_tpr" if label else "any_fpr")
                        if route == "*"
                        else ("tpr" if label else "fpr"),
                        "n": len(ids),
                        "fully_scored": int(scored.sum()),
                        "gain": delta,
                        "lower": lo,
                        "upper": hi,
                    }
                )
    return out


def recording_outcomes(panel, policy):
    """Small keyed arrays for secondary inference retaining all linked recordings."""
    d = flags(panel, policy)
    missing = ~np.isfinite(panel.scores)
    return {
        "recording": panel.meta.recording_id.to_numpy(),
        "cluster": panel.meta.cluster_id.to_numpy(),
        "genuine": panel.real,
        "available": ~missing.any(axis=1),
        "any_fpr": d.any(axis=1),
        "all_tpr": d.all(axis=1),
        "native_tpr": d[:, 0],
    }


def cluster_contrasts(cases, primary, comparators, cohorts, draws):
    """Recording-weighted paired differences with lineage-block bootstrap percentile intervals.

    Sample whole independent clusters with replacement, retaining all siblings.
    Paired counts/denominators are summed over selected cluster blocks before division.
    No frame, route or individual sibling is sampled independently.
    """
    if not draws:
        return []
    rng = np.random.default_rng(465932)
    results = []
    for cohort in cohorts:
        a = cases.get((primary, cohort))
        if a is None:
            continue
        for name, run in comparators.items():
            b = cases.get((run, cohort))
            if b is None:
                continue
            if not np.array_equal(a["recording"], b["recording"]):
                raise ValueError(
                    "Recording-weighted contrast misaligns the matched recordings"
                )
            for metric in ("any_fpr", "all_tpr", "native_tpr"):
                mask = a["genuine"] if metric == "any_fpr" else ~a["genuine"]
                if not mask.any():
                    continue
                complete = a["available"][mask] & b["available"][mask]
                groups, ids = np.unique(a["cluster"][mask], return_inverse=True)
                point, lo, hi = (np.nan,) * 3
                if complete.all():
                    d = a[metric][mask].astype(int) - b[metric][mask].astype(int)
                    point = float(d.mean())
                    if len(groups) > 1:
                        den = np.bincount(ids)
                        num = np.bincount(ids, weights=d)
                        dist = np.empty(draws)
                        for i in range(draws):
                            ix = rng.integers(len(groups), size=len(groups))
                            dist[i] = num[ix].sum() / den[ix].sum()
                        lo, hi = np.quantile(dist, [0.025, 0.975])
                results.append(
                    {
                        "cohort": cohort,
                        "comparison": name,
                        "comparator_run": run,
                        "metric": metric,
                        "n_recordings": int(mask.sum()),
                        "n_clusters": len(groups),
                        "fully_scored": int(complete.sum()),
                        "difference": point,
                        "percentile_lower": lo,
                        "percentile_upper": hi,
                        "replicates": draws,
                    }
                )
    return results


def session_rates(joined, run, policy, cohort, test):
    """Describe genuine/fake route rates within each measured receiver session."""
    fields = [
        "capture_method",
        "capture_channel",
        "capture_session",
        "capture_settings",
    ]
    if "kind" not in joined or not set(fields) <= set(joined):
        return []
    observed = joined[
        (joined.cohort == cohort)
        & (joined.split == "test")
        & (joined.kind == "measured")
    ].copy()
    keys = test.meta.recording_id
    # One prespecified independent recording representative within each class,
    # preserved across routes and detectors; sample sizes vary by capture session.
    observed = observed[observed.recording_id.isin(keys)]
    if observed.empty:
        return []
    index = dict(zip(keys, range(len(keys))))
    route_ix = {k: j for j, k in enumerate(test.routes)}
    i = observed.recording_id.map(index).to_numpy(int)
    j = observed.route.map(route_ix).to_numpy(int)
    decisions = flags(test, policy)[i, j]
    known = np.isfinite(test.scores[i, j])
    observed["event"] = np.where(
        observed.label == 0, decisions | ~known, decisions & known
    )
    observed["scored"] = known
    rows = []
    for key, group in observed.groupby(["route", *fields, "label"], dropna=False):
        cases = len(group)
        count = int(group.event.sum())
        rows.append(
            {
                "cohort": cohort,
                "run": run,
                **dict(zip(["route", *fields, "label"], key)),
                "metric": "tpr" if int(key[-1]) else "fpr",
                "n": cases,
                "n_scored": int(group.scored.sum()),
                "events": count,
                "estimate": count / cases,
            }
        )
    return rows
