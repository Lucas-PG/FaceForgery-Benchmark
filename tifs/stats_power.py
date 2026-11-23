"""Design-stage power screen using independent lineages, without test scores."""

import math

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

from .stats_core import paired_count_bounds


def _paired_power(n, p_plus, p_minus, margin, tail, draws, rng):
    if n <= 0:
        return 0.0
    if p_plus < 0 or p_minus < 0 or p_plus + p_minus > 1:
        raise ValueError("Invalid assumed paired discordance probabilities")
    plus, minus, _ = rng.multinomial(
        n, [p_plus, p_minus, 1 - p_plus - p_minus], size=draws
    ).T
    lo, _ = paired_count_bounds(plus, minus, n, tail)
    return float(np.mean(lo > margin))


def _ceiling_power(
    n_genuine, n_fake, rank, tail, alpha, frozen_recall, learned_recall, draws, rng
):
    """Optimistic reference design: identical within-recording route ranks; nested fake responses.

    This allows a genuinely attainable marginal threshold ceiling. The baseline fake
    score CDF is x**shape on [0,1] and the genuine CDF is uniform. Scores used to
    estimate the relaxed cutoff are independent of the later paired fake cases.
    """
    if not n_genuine or not n_fake or not rank:
        return 0.0
    floor = rng.beta(rank, n_genuine + 1 - rank, draws)
    shape = np.log1p(-frozen_recall) / np.log1p(-alpha)
    relaxed = 1 - floor**shape
    # Nested responses via one uniform latent rank per test forgery. Conditional
    # binomial sampling avoids allocating draws × n_fake underlying uniforms.
    shared = np.minimum(relaxed, learned_recall)
    both = rng.binomial(n_fake, shared)
    one = rng.binomial(
        n_fake - both, np.clip(abs(relaxed - learned_recall) / (1 - shared), 0, 1)
    )
    learned = both + np.where(learned_recall > relaxed, one, 0)
    frozen = both + np.where(relaxed > learned_recall, one, 0)
    lo = np.zeros(draws)
    lo[learned > 0] = beta.ppf(
        tail, learned[learned > 0], n_fake - learned[learned > 0] + 1
    )
    hi = np.ones(draws)
    hi[frozen < n_fake] = beta.isf(
        tail / 2, frozen[frozen < n_fake] + 1, n_fake - frozen[frozen < n_fake]
    )
    return float(np.mean(lo > hi))


def confirmation_cohorts(cfg):
    """Return the prospectively named primary cohorts; other registered targets are transport only."""
    opt = cfg["analysis"]
    target = opt["cohorts"]
    active = opt.get("confirmation_cohorts", target)
    minimum = 2 if cfg.get("mode") == "confirmatory" else 1
    if (
        not isinstance(active, list)
        or len(active) < minimum
        or len(set(active)) != len(active)
        or not set(active) <= set(target)
    ):
        raise ValueError(
            "confirmation_cohorts must be a nonempty distinct subset, with at least two for confirmation"
        )
    return active


def power_audit(cfg, panel):
    """Projected gate power under explicit planning scenarios, never under fitted test rates."""
    opt = cfg["analysis"]
    alpha = float(opt.get("alpha", 0.05))
    confidence = float(opt.get("confidence", 0.95))
    cohorts = opt["cohorts"]
    confirm = confirmation_cohorts(cfg)
    n_comparisons = 5 + len(opt.get("control_runs", []))
    tail = (1 - confidence) / (
        2 * len(cohorts) * (n_comparisons + 18 + 10 * len(opt.get("replications", [])))
    )
    scenarios = opt.get("power", {})
    guard = float(scenarios.get("true_fpr", 0.6 * alpha))
    expected_gain = scenarios.get("gain_discordance", [0.15, 0.05])
    expected_clean = scenarios.get("clean_discordance", [0.05, 0.05])
    frozen_recall = float(scenarios.get("frozen_ceiling", 0.30))
    learned_recall = float(scenarios.get("learned_recall", 0.50))
    if not 0 < frozen_recall < 1 or not 0 < learned_recall < 1:
        raise ValueError("Invalid assumed ceiling scenario")
    draws = int(scenarios.get("draws", 4000))
    if not 0 < guard < alpha or not 100 <= draws <= 100000:
        raise ValueError("Invalid predeclared design power assumptions")
    rng = np.random.default_rng(991)
    result = []
    for cohort in cohorts:
        test = panel[(panel.split == "test") & (panel.cohort.astype(str) == cohort)]
        diag = panel[
            (panel.split == "diagnostic") & (panel.cohort.astype(str) == cohort)
        ]
        real = int(test.loc[test.label.astype(int) == 0, "cluster_id"].nunique())
        fake = int(test.loc[test.label.astype(int) == 1, "cluster_id"].nunique())
        diagnostic = int(diag.loc[diag.label.astype(int) == 0, "cluster_id"].nunique())
        # Largest k with P_{p=alpha}(X<=k) <= tail.
        critical = int(binom.ppf(tail, real, alpha)) if real else -1
        if real and binom.cdf(critical, real, alpha) > tail:
            critical -= 1
        routes = test.route.nunique()
        rank = (
            int(binom.ppf(tail / (2 * routes), diagnostic, 1 - alpha))
            if diagnostic and routes
            else 0
        )
        result.append(
            {
                "cohort": cohort,
                "role": "confirmation" if cohort in confirm else "transport",
                "n_genuine": real,
                "n_fake": fake,
                "n_diagnostic_genuine": diagnostic,
                "confirmatory_tail": tail,
                "maximum_passing_false_alerts": critical,
                "diagnostic_quantile_rank": rank,
                "minimum_genuine_for_zero_flags": math.ceil(
                    math.log(tail) / math.log1p(-alpha)
                ),
                # With no observed native discordances the interval is [-u,u],
                # where u=1-(tail/2)**(1/n). Noninferiority is strictly above -margin.
                "n_fake_native_zero_discordance": (
                    math.floor(
                        math.log(tail / 2)
                        / math.log1p(-float(opt.get("clean_margin", 0.02)))
                    )
                    + 1
                    if float(opt.get("clean_margin", 0.02)) > 0
                    else None
                ),
                # All n comparisons are wins. Lower bound x*(2*x-1) for x>1/2,
                # where x=(tail/2)**(1/n), must strictly exceed the gain margin.
                "minimum_fake_if_every_case_improves": math.floor(
                    math.log(tail / 2)
                    / math.log(
                        (1 + math.sqrt(1 + 8 * float(opt.get("gain_margin", 0.05)))) / 4
                    )
                )
                + 1,
                "assumed_true_fpr": guard,
                "fpr_power": float(binom.cdf(critical, real, guard)) if real else 0.0,
                "assumed_gain_plus": expected_gain[0],
                "assumed_gain_minus": expected_gain[1],
                "gain_power": _paired_power(
                    fake,
                    *expected_gain,
                    float(opt.get("gain_margin", 0.05)),
                    tail,
                    draws,
                    rng,
                ),
                "assumed_clean_plus": expected_clean[0],
                "assumed_clean_minus": expected_clean[1],
                "clean_power": _paired_power(
                    fake,
                    *expected_clean,
                    -float(opt.get("clean_margin", 0.02)),
                    tail,
                    draws,
                    rng,
                ),
                "assumed_frozen_ceiling": frozen_recall,
                "assumed_learned_recall": learned_recall,
                "reference_ceiling_power": _ceiling_power(
                    diagnostic,
                    fake,
                    rank,
                    tail,
                    alpha,
                    frozen_recall,
                    learned_recall,
                    draws,
                    rng,
                ),
                "simulated_draws": draws,
            }
        )
    return pd.DataFrame(result)
