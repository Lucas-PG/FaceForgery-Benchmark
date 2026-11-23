"""Synthetic checks exercise normal server entry points, independent of torch/video I/O."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, multinomial

from tifs.analysis import analyze
from tifs.stats_core import (
    auc,
    ceiling,
    cutoff_scores,
    fit_policies,
    flags,
    interval,
    paired_count_bounds,
    paired_difference,
    representatives,
    wide,
)


def make_panel(split="dev", n=30, cohort="source", routes=("native", "codec")):
    rows = []
    rng = np.random.default_rng(826)
    for i in range(n):
        for label in (0, 1):
            item = f"{split}/{cohort}/{i}/{label}"
            for j, route in enumerate(routes):
                rows.append(
                    {
                        "recording_id": item,
                        "cluster_id": f"{split}/{cohort}/{i}",
                        "route": route,
                        "cohort": cohort,
                        "family": "swap" if label else "genuine",
                        "split": split,
                        "label": label,
                        "score": float(rng.normal() + 0.5 * label + 0.3 * j),
                        "quality": float(i * j % 7),
                    }
                )
    return pd.DataFrame(rows)


class StatisticsTests(unittest.TestCase):
    def test_measures_same_cases_and_ties(self):
        s = np.array([[0.8, 0.1], [0.1, 0.8], [0.8, 0.8], [0.5, 0.5]])
        spec = {"kind": "shared", "cutoff": 0.5}
        v = wide(make_panel(n=2))
        d = flags(type(v)(v.meta, v.routes, s, v.quality), spec)
        self.assertEqual(int(d.all(axis=1).sum()), 1)
        self.assertEqual(int(d.any(axis=1).sum()), 3)
        self.assertEqual(float(d.sum(axis=0).min() / 4), 0.5)

    def test_auc_and_paired_interval(self):
        self.assertEqual(auc([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9]), 1)
        self.assertEqual(auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(interval(0, 20, 0.025)[0], 0)
        a = np.array([True, True, False, True])
        b = np.array([True, False, False, False])
        delta, lo, hi = paired_difference(a, b, 0.025)
        self.assertEqual(delta, 0.5)
        self.assertLess(lo, delta)
        self.assertGreater(hi, delta)

    def test_paired_interval_enumerated_finite_sample_coverage(self):
        n, tail = 16, 0.05
        cells = np.asarray(
            [(i, j, n - i - j) for i in range(n + 1) for j in range(n - i + 1)]
        )
        lo, hi = paired_count_bounds(cells[:, 0], cells[:, 1], n, tail)
        for discordance in (0.01, 0.15, 0.5, 0.99, 1.0):
            for win in (0.0, 0.04, 0.4, 0.5, 0.85, 1.0):
                p = discordance * win
                q = discordance * (1 - win)
                masses = multinomial.pmf(cells, n, [p, q, 1 - discordance])
                self.assertLessEqual(masses[lo > p - q + 1e-12].sum(), tail + 1e-12)
                self.assertLessEqual(masses[hi < p - q - 1e-12].sum(), tail + 1e-12)
        # Identical observed results do not establish exact population equality.
        lower, upper = paired_count_bounds(0, 0, n, tail)
        self.assertAlmostEqual(float(upper), 1 - (tail / 2) ** (1 / n))
        self.assertAlmostEqual(float(lower), -float(upper))

    def test_ceiling_is_conservative_and_diagnostic_disjoint(self):
        d = representatives(wide(make_panel("diagnostic", n=80)))
        t = representatives(wide(make_panel("test", n=120)))
        c = ceiling(d, t, 0.1, 0.01)
        self.assertLessEqual(c["estimate"], c["upper"])
        k = c["order_rank"]
        self.assertLessEqual(binom.cdf(k - 1, 80, 0.9), 0.01 / (2 * len(d.routes)))
        with self.assertRaisesRegex(ValueError, "share lineages"):
            ceiling(d, d, 0.1, 0.01)

    def test_ceiling_bounds_all_feasible_threshold_pairs_with_ties(self):
        from itertools import product

        diag = wide(make_panel("diagnostic", n=8))
        test = wide(make_panel("test", n=9))
        diag = replace(diag, scores=diag.scores.copy())
        test = replace(test, scores=test.scores.copy())
        real = np.array(
            [[0, 0], [0, 1], [1, 0], [0, 0], [1, 1], [2, 1], [1, 2], [1, 1]], float
        )
        fake = np.array(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]],
            float,
        )
        diag.scores[diag.real] = real
        test.scores[~test.real] = fake
        result = ceiling(diag, test, 0.25, 0.025)
        achievable = max(
            (fake > cuts).all(1).mean()
            for cuts in product([-1.0, 0.0, 1.0, 2.0], repeat=2)
            if (real > cuts).any(1).mean() <= 0.25
        )
        self.assertLessEqual(achievable, result["estimate"])
        self.assertLessEqual(result["estimate"], result["upper"])

    def test_upper_bound_coverage_on_known_population(self):
        # Uniform independent genuine scores; independent fake route scores have CDF x^2.
        # The exact joint marginal-floor upper bound is (1-(1-alpha)^2)^2.
        rng = np.random.default_rng(31415)
        d = wide(make_panel("diagnostic", n=46))
        t = wide(make_panel("test", n=74))
        d = replace(d, scores=d.scores.copy())
        t = replace(t, scores=t.scores.copy())
        truth = (1 - 0.8**2) ** 2
        upper = []
        for _ in range(250):
            d.scores[d.real] = rng.random((46, 2))
            t.scores[~t.real] = np.sqrt(rng.random((74, 2)))
            upper.append(ceiling(d, t, 0.2, 0.05)["upper"])
        # Check gross implementation/calculation regressions, not a proof of coverage.
        self.assertGreaterEqual(np.mean(np.array(upper) >= truth), 0.94)

    def test_guarded_cutoff_and_fallback(self):
        dev = wide(make_panel(n=50))
        cal = wide(make_panel("cal", n=50))
        p = fit_policies(dev, cal, 0.05)
        self.assertEqual(
            p["shared"]["cutoff"],
            np.quantile(cal.scores[cal.real].max(1), 0.95, method="higher"),
        )
        target = wide(make_panel("test", routes=("native", "new")))
        self.assertEqual(flags(target, p["route"]).shape, target.scores.shape)
        np.testing.assert_array_equal(
            flags(target, p["marginal"])[:, 1],
            target.scores[:, 1] > max(p["marginal"]["cutoff"].values()),
        )
        self.assertEqual(
            cutoff_scores(np.ones((3, 2)), "shared", 0.05, ("native", "a")), 1.0
        )

    def test_duplicate_views_and_incomplete_panels(self):
        r = make_panel()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            wide(r.iloc[1:])
        with self.assertRaisesRegex(ValueError, "duplicate routes"):
            wide(pd.concat([r, r.iloc[[0]]]))

    def test_choose_representative_without_scores_or_row_order(self):
        r = make_panel()
        duplicate = r.copy()
        duplicate.recording_id += "/b"
        duplicate.score = 999
        complete = pd.concat([r, duplicate])
        ids = lambda x: list(representatives(wide(x)).meta.recording_id)
        self.assertEqual(ids(complete), ids(complete.sample(frac=1, random_state=2)))
        self.assertEqual(len(ids(complete)), len(r) // 2)

    def test_transport_failure_cannot_reverse_confirmatory_cohort_claim(self):
        # Three independent targets. The same deployed model intentionally fails on
        # the third; this must remain visible without changing two preregistered roles.
        names = ["pair__s42", "aug__s42", "shuffle__s42", "clean__s42"]
        roles = dict(
            zip(("primary_run", "baseline_run", "shuffled_run", "clean_run"), names)
        )
        cfg = {
            "mode": "confirmatory",
            "analysis": {
                **roles,
                "bootstrap": 0,
                "cohorts": ["a", "b", "transfer"],
                "confirmation_cohorts": ["a", "b"],
            },
        }
        from tifs.stats_power import confirmation_cohorts

        with self.assertRaisesRegex(ValueError, "at least two"):
            confirmation_cohorts(
                {
                    "mode": "confirmatory",
                    "analysis": dict(cfg["analysis"], confirmation_cohorts=["a"]),
                }
            )
        frames = [make_panel("dev", n=20), make_panel("cal", n=20)]
        for cohort in cfg["analysis"]["cohorts"]:
            for split in ["diagnostic", "test"]:
                frames.append(make_panel(split, n=500, cohort=cohort))
        panel = pd.concat(frames, ignore_index=True).drop(columns="score")
        panel["kind"] = np.where(panel.route.eq("native"), "native", "measured")
        for field in (
            "capture_session",
            "capture_method",
            "capture_channel",
            "capture_settings",
        ):
            panel[field] = "fixed"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            predictions = {}
            for name in names:
                scores = (
                    panel.label.to_numpy().copy()
                    if name in {names[0], names[-1]}
                    else np.zeros(len(panel))
                )
                if name == names[0]:
                    m = panel.cohort.eq("transfer")
                    scores[m] = 1 - scores[m]
                predictions[name] = p / (name + ".csv")
                panel[["recording_id", "route"]].assign(score=scores).to_csv(
                    predictions[name], index=False
                )
            out = p / "analysis"
            result = analyze(cfg, panel, predictions, out)
            self.assertEqual(result["training_claim"], "supported")
            self.assertEqual(result["cohorts"]["transfer"], "contradicted")
            self.assertTrue(result["all_seed_directions_consistent"])
            cov = pd.read_csv(out / "coverage.csv")
            self.assertEqual(set(cov[cov.cohort == "transfer"].role), {"transport"})
            session = pd.read_csv(out / "capture_sessions.csv")
            self.assertEqual(
                set(
                    session.query(
                        "run == 'pair__s42' and cohort == 'transfer' and label == 0"
                    ).estimate
                ),
                {1.0},
            )
            self.assertEqual(
                set(
                    session.query(
                        "run == 'pair__s42' and cohort == 'a' and label == 0"
                    ).estimate
                ),
                {0.0},
            )

    def test_complete_analysis_and_no_external_tuning(self):
        a = make_panel("dev", n=32)
        b = make_panel("cal", n=20)
        d = make_panel(
            "diagnostic", n=15, cohort="external", routes=("native", "delivered")
        )
        e = make_panel("test", n=18, cohort="external", routes=("native", "delivered"))
        whole = pd.concat([a, b, d, e], ignore_index=True)
        specs = [
            "paired__s42",
            "erm__s42",
            "shuffle__s42",
            "clean__s42",
            "temporal__s42",
            "b-paired__s42",
            "b-erm__s42",
            "b-shuffle__s42",
            "b-clean__s42",
            "paired__s123",
            "erm__s123",
            "shuffle__s123",
            "clean__s123",
        ]
        cfg = {
            "analysis": {
                "primary_run": specs[0],
                "baseline_run": specs[1],
                "shuffled_run": specs[2],
                "clean_run": specs[3],
                "cohorts": ["external"],
                "bootstrap": 4,
                "heldout_families": {"external": ["swap"]},
                "replications": [
                    dict(
                        zip(
                            (
                                "primary_run",
                                "baseline_run",
                                "shuffled_run",
                                "clean_run",
                            ),
                            specs[5:9],
                        )
                    )
                ],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            preds = {}
            for n, name in enumerate(specs):
                x = whole[["recording_id", "route", "score"]].copy()
                x.score += n * 0.2
                if n == 4:
                    x["failure"] = ""
                    x.loc[x.index[-1], ["score", "failure"]] = [np.nan, "no_face"]
                preds[name] = p / f"{name}.csv"
                x.to_csv(preds[name], index=False)
            panel = whole.drop(columns="score")
            first = analyze(cfg, panel, preds, p / "a")
            self.assertTrue(first["training_claim"] in {"inconclusive", "contradicted"})
            self.assertEqual(len(pd.read_csv(p / "a" / "seed_contrasts.csv")), 3)
            self.assertEqual(len(pd.read_csv(p / "a" / "replications.csv")), 1)
            rows = pd.read_csv(p / "a" / "subgroup_contrasts.csv")
            delivered = rows.query(
                "comparison == 'augmented' and route == 'delivered' and family == 'swap'"
            )
            self.assertEqual(len(delivered), 1)
            self.assertEqual(delivered.iloc[0].exposure, "held_out_route")
            self.assertAlmostEqual(delivered.iloc[0].gain, 0.0)
            self.assertEqual(set(rows.query("family == 'swap'").label), {1})
            clusters = pd.read_csv(p / "a" / "cluster_sensitivity.csv")
            self.assertTrue((clusters.difference == 0).all())
            contrasts = pd.read_csv(p / "a" / "contrasts.csv")
            self.assertEqual(
                contrasts.query("comparison == 'matched_baseline'").iloc[0].status,
                "reference_eligibility_unverified",
            )
            self.assertEqual(len(contrasts.query("metric == 'reference_any_fpr'")), 2)
            mixture = pd.read_csv(p / "a" / "route_mixture.csv")
            one = mixture.query(
                "run == 'paired__s42' and policy == 'shared' and metric == 'mean_tpr'"
            ).iloc[0]
            self.assertEqual(one.n_recordings, 18)
            marginals = pd.read_csv(p / "a" / "metrics.csv").query(
                "run == 'paired__s42' and policy == 'shared' and metric == 'tpr' and family == '*'"
            )
            self.assertAlmostEqual(one.estimate, marginals.estimate.mean())
            self.assertTrue(one.lower <= one.estimate <= one.upper)
            power = pd.read_csv(p / "a" / "power.csv")
            self.assertTrue(power.reference_ceiling_power.between(0, 1).all())
            self.assertEqual(len(pd.read_csv(p / "a" / "ceiling.csv")), 2)
            self.assertFalse(first["all_seed_directions_consistent"])
            metrics = pd.read_csv(p / "a" / "metrics.csv")
            v = metrics.query(
                "run == 'temporal__s42' and policy == 'shared' and metric == 'all_tpr' and family == '*'"
            ).iloc[0]
            self.assertEqual(v.n, 18)
            self.assertEqual(v.n_fully_scored, 17)
            self.assertEqual(len(pd.read_csv(p / "a" / "calibration_bootstrap.csv")), 1)
            policies = json.loads((p / "a" / "policies.json").read_text())
            # Permute and shift ALL external scores for all detectors. Fitting and source
            # model selection must remain byte-identical, but external estimates may change.
            for path in preds.values():
                x = pd.read_csv(path, float_precision="round_trip")
                mask = x.recording_id.str.startswith(("test/", "diagnostic/"))
                x.loc[mask, "score"] = 100 - x.loc[mask, "score"]
                x.to_csv(path, index=False)
            analyze(cfg, panel, preds, p / "b")
            self.assertEqual(
                policies, json.loads((p / "b" / "policies.json").read_text())
            )
            self.assertEqual(
                (p / "a" / "development_selection.csv").read_bytes(),
                (p / "b" / "development_selection.csv").read_bytes(),
            )
            bad = panel.copy()
            bad.loc[bad.split == "test", "cluster_id"] = panel.loc[
                panel.split == "cal", "cluster_id"
            ].iloc[0]
            with self.assertRaisesRegex(ValueError, "crosses"):
                analyze(cfg, bad, preds, p / "bad")


if __name__ == "__main__":
    unittest.main()
