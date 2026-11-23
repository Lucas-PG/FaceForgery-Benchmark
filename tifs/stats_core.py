"""Recording-level decision rules. An alert always means score strictly above cutoff."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta, binom, rankdata


def digest_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\x00{value}".encode()).hexdigest()


@dataclass(frozen=True)
class Views:
    meta: pd.DataFrame
    routes: tuple[str, ...]
    scores: np.ndarray
    quality: np.ndarray

    def take(self, indexes) -> Views:
        idx = np.asarray(indexes)
        return Views(
            self.meta.iloc[idx].reset_index(drop=True),
            self.routes,
            self.scores[idx],
            self.quality[idx],
        )

    @property
    def real(self):
        return self.meta["label"].to_numpy() == 0


def wide(rows: pd.DataFrame) -> Views:
    if rows.empty:
        raise ValueError("Empty recording panel")
    keys = ["recording_id", "route"]
    if rows.duplicated(keys).any():
        raise ValueError("A recording has duplicate routes")
    metadata = ["cohort", "split", "family", "label", "cluster_id"]
    if (rows.groupby("recording_id")[metadata].nunique(dropna=False) != 1).any().any():
        raise ValueError("Renditions have contradictory recording metadata")
    routes = tuple(sorted(rows.route.unique(), key=lambda r: (r != "native", r)))
    if routes[0] != "native":
        raise ValueError("Every panel needs a native rendition")
    indexed = rows.set_index(keys)
    scores = indexed.score.unstack().loc[:, list(routes)]
    quality = indexed.quality.unstack().loc[scores.index, list(routes)]
    if (
        len(rows) != len(scores) * len(routes)
        or np.isinf(scores.to_numpy()).any()
        or not np.isfinite(quality.to_numpy()).all()
    ):
        raise ValueError(
            "Incomplete paired panel or nonfinite scores/quality; inspect attrition before inference"
        )
    meta = (
        rows.drop_duplicates("recording_id")
        .set_index("recording_id")
        .loc[scores.index]
        .reset_index()
    )
    return Views(
        meta[["recording_id", *metadata]],
        routes,
        scores.to_numpy(float),
        quality.to_numpy(float),
    )


def representatives(panel: Views) -> Views:
    """One deterministic representative per independent lineage and label, chosen without scores."""
    order = panel.meta.recording_id.map(
        lambda v: digest_key("tifs-representative-v1", v)
    )
    subset = (
        panel.meta.assign(_order=order)
        .sort_values("_order")
        .drop_duplicates(["cluster_id", "label"])
    )
    return panel.take(np.sort(subset.index.to_numpy()))


def split_dev(panel: Views) -> tuple[Views, Views]:
    # Split lineage components, balancing the presence of each class where possible.
    signatures = panel.meta.groupby("cluster_id").label.apply(
        lambda v: tuple(sorted(set(v)))
    )
    chosen = set()
    for _, groups in signatures.groupby(signatures):
        ordered = sorted(
            groups.index, key=lambda v: digest_key("tifs-dev-selection-v1", v)
        )
        chosen.update(ordered[::2])
    selector = panel.meta.cluster_id.isin(chosen).to_numpy()
    fit, select = panel.take(~selector), panel.take(selector)
    for sub in (fit, select):
        if sub.meta.label.nunique() != 2:
            raise ValueError(
                "dev needs independent lineages of both labels in fit and selection folds"
            )
    return fit, select


def interval(events: int, n: int, tail: float) -> tuple[float, float]:
    """Clopper–Pearson, tail probability per side; both endpoints inclusive."""
    if not 0 <= events <= n or n <= 0 or not 0 < tail < 0.5:
        raise ValueError("Invalid binomial interval")
    lo = 0.0 if events == 0 else float(beta.ppf(tail, events, n - events + 1))
    hi = 1.0 if events == n else float(beta.isf(tail, events + 1, n - events))
    return lo, hi


def paired_count_bounds(plus, minus, n, tail):
    """Exact factorized interval for a paired difference, including zero discordances.

    With discordance probability d and conditional win probability w, the signed
    difference is d*(2*w-1). The discordance count is binomial(n,d); conditional
    on this count the number of wins is binomial(discordances,w). A joint confidence
    rectangle uses two Clopper–Pearson intervals. Taking its extrema is valid even
    when the sign of the difference is unknown or the two cells have equal counts.
    """
    plus, minus = np.broadcast_arrays(np.asarray(plus), np.asarray(minus))
    total = plus + minus
    if n <= 0 or not 0 < tail < 0.5 or np.any((plus < 0) | (minus < 0) | (total > n)):
        raise ValueError("Invalid paired discordance counts")

    def bounds(k, trials):
        trials = np.broadcast_to(trials, k.shape)
        lo, hi = np.zeros(k.shape), np.ones(k.shape)
        select = k > 0
        lo[select] = beta.ppf(tail / 2, k[select], trials[select] - k[select] + 1)
        select = k < trials
        hi[select] = beta.isf(tail / 2, k[select] + 1, trials[select] - k[select])
        return lo, hi

    dl, du = bounds(total, n)
    wl, wu = bounds(plus, total)
    sign_lo, sign_hi = 2 * wl - 1, 2 * wu - 1
    return np.minimum(dl * sign_lo, du * sign_lo), np.maximum(
        dl * sign_hi, du * sign_hi
    )


def paired_difference(first, second, tail):
    """Difference and finite-sample interval on matched, independent recording units."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if len(a) == 0 or a.shape != b.shape:
        raise ValueError("Invalid matched contrast")
    plus = int((a & ~b).sum())
    minus = int((~a & b).sum())
    lower, upper = paired_count_bounds(plus, minus, len(a), tail)
    return (plus - minus) / len(a), float(lower), float(upper)


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    n = int(labels.sum())
    m = len(labels) - n
    if min(n, m) == 0:
        raise ValueError("AUROC requires both classes")
    return float((rankdata(scores)[labels].sum() - n * (n + 1) / 2) / (n * m))


def quantile(x, alpha, axis=None):
    return np.quantile(x, 1 - alpha, axis=axis, method="higher")


def quality_features(s, q, center, scale):
    x = (s - center[0]) / scale[0]
    y = (q - center[1]) / scale[1]
    return np.stack([x, y, x * y], axis=-1)


def fit_quality(panel: Views) -> dict:
    s, q = panel.scores.ravel(), panel.quality.ravel()
    y = np.repeat(panel.meta.label.to_numpy(), len(panel.routes))
    center = [float(np.mean(s)), float(np.mean(q))]
    scale = [max(float(np.std(s)), 1e-8), max(float(np.std(q)), 1e-8)]
    x = quality_features(s, q, center, scale)

    def cost(w):
        z = x @ w[:3] + w[3]
        loss = np.logaddexp(0.0, z).mean() - np.mean(y * z) + 0.005 * (w[:3] @ w[:3])
        d = expit(z) - y
        return loss, np.r_[x.T @ d / len(y) + 0.01 * w[:3], d.mean()]

    fit = minimize(cost, np.zeros(4), jac=True, method="L-BFGS-B")
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Quality fit failed: {fit.message}")
    return {"center": center, "scale": scale, "weights": fit.x.tolist()}


def fit_offsets(dev: Views, alpha: float) -> dict:
    """Coordinate search on dev only, computing a joint genuine cutoff for every candidate.

    Offsets allocate the alert budget. The eventual global cutoff is re-estimated on cal.
    This is an attainable finite-grid comparator, not a bound on the oracle.
    """
    real, fake = dev.scores[dev.real], dev.scores[~dev.real]
    offsets = np.zeros(len(dev.routes))

    def criterion(value):
        threshold = quantile((real - value).max(axis=1), alpha)
        return int(np.sum((fake - value).min(axis=1) > threshold))

    best = criterion(offsets)
    for _ in range(3):
        changed = False
        for j in range(1, len(offsets)):
            # Include symmetric differences of plausible per-route upper-tail cutoffs.
            qs = np.linspace(max(0.0, 1 - 3 * alpha), 1.0, 15)
            ref = np.quantile(real[:, 0], qs, method="higher")
            cur = np.quantile(real[:, j], qs, method="higher")
            candidates = np.unique((cur[:, None] - ref).ravel())
            incumbent = offsets[j]
            for candidate in candidates:
                trial = offsets.copy()
                trial[j] = candidate
                score = criterion(trial)
                if score > best:
                    incumbent, best, changed = candidate, score, True
            offsets[j] = incumbent
        if not changed:
            break
    return dict(zip(dev.routes, offsets.tolist()))


def transform(panel: Views, policy: dict):
    if policy["kind"] == "quality":
        fit = policy["quality_fit"]
        x = quality_features(panel.scores, panel.quality, fit["center"], fit["scale"])
        w = np.array(fit["weights"])
        return x @ w[:3] + w[3]
    if policy["kind"] == "route":
        offsets = policy["offsets"]
        # An unseen route has no learned offset. Use the most conservative learned offset.
        return panel.scores - [
            offsets.get(r, max(offsets.values())) for r in panel.routes
        ]
    return panel.scores


def cutoff_scores(real_scores: np.ndarray, kind: str, alpha: float, routes):
    if kind == "marginal":
        return dict(zip(routes, quantile(real_scores, alpha, axis=0).tolist()))
    if kind == "native":
        return float(quantile(real_scores[:, routes.index("native")], alpha))
    return float(quantile(real_scores.max(axis=1), alpha))


def fit_policies(dev: Views, cal: Views, source_alpha: float) -> dict:
    if not 0 < source_alpha < 1:
        raise ValueError("Invalid source calibration rate")
    if dev.routes != cal.routes:
        raise ValueError(
            "Source development and calibration panels must contain the same routes"
        )
    if dev.real.all() or not dev.real.any() or not cal.real.any():
        raise ValueError(
            "Calibration needs genuine recordings and development needs both classes"
        )
    policies = {
        name: {"kind": name}
        for name in ("shared", "route", "quality", "native", "marginal")
    }
    policies["quality"]["quality_fit"] = fit_quality(dev)
    policies["route"]["offsets"] = fit_offsets(dev, source_alpha)
    for p in policies.values():
        real = transform(cal, p)[cal.real]
        p["cutoff"] = cutoff_scores(real, p["kind"], source_alpha, cal.routes)
    return policies


def flags(panel: Views, policy: dict):
    threshold = policy["cutoff"]
    if isinstance(threshold, dict):
        threshold = [threshold.get(r, max(threshold.values())) for r in panel.routes]
    return transform(panel, policy) > threshold


def ceiling(diag: Views, test: Views, target_alpha: float, tail: float) -> dict:
    """Upper confidence limit for ANY deterministic monotone within-route threshold rule."""
    if diag.routes != test.routes:
        raise ValueError(
            "Diagnostic and sealed target panels must have identical route sets"
        )
    if set(diag.meta.cluster_id) & set(test.meta.cluster_id):
        raise ValueError("Diagnostic quantiles and test observations share lineages")
    real = diag.scores[diag.real]
    fake = test.scores[~test.real]
    if not min(len(real), len(fake)):
        raise ValueError("Ceiling needs diagnostic genuine and test fake lineages")
    k = int(binom.ppf(tail / (2 * len(diag.routes)), len(real), 1 - target_alpha))
    # P(X_(k) > q) <= P(Bin(n,1-alpha) < k). k=0 is a vacuous lower bound.
    floor = np.sort(real, axis=0)[k - 1] if k else np.full(len(diag.routes), -np.inf)
    passing = (fake > floor).all(axis=1)
    # The empirical CDF inverse, unlike "higher" interpolated sample quantiles,
    # is the least feasible marginal cutoff when the sample itself is the population.
    empirical_floor = np.quantile(real, 1 - target_alpha, axis=0, method="inverted_cdf")
    plug = (fake > empirical_floor).all(axis=1)
    return {
        "estimate": float(plug.mean()),
        "upper": interval(int(passing.sum()), len(fake), tail / 2)[1],
        "relaxed_events": int(passing.sum()),
        "n_fake": len(fake),
        "n_genuine": len(real),
        "order_rank": k,
        "lower_quantiles": {
            r: (float(v) if np.isfinite(v) else None)
            for r, v in zip(diag.routes, floor)
        },
    }
