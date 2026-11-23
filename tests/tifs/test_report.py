import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tifs.analysis import analyze
from tifs.io import sha, write_json
from tifs.report import report


class ReportTests(unittest.TestCase):
    def test_report_tracks_source_artifacts_and_preserves_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            rng = np.random.default_rng(42)
            for split in ("dev", "cal", "diagnostic", "test"):
                cohort = "held_out" if split in ("diagnostic", "test") else "source"
                for i in range(10):
                    for label in range(2):
                        for r in ("native", "encoded"):
                            records.append(
                                {
                                    "cohort": cohort,
                                    "split": split,
                                    "cluster_id": f"{split}:{i}",
                                    "recording_id": f"{split}:{i}:{label}",
                                    "route": r,
                                    "label": label,
                                    "family": "swap" if label else "authentic",
                                    "quality": rng.random(),
                                }
                            )
            panel = pd.DataFrame(records)
            names = ["paired__s42", "aug__s42", "shuffled__s42", "clean__s42"]
            roles = dict(
                zip(("primary_run", "baseline_run", "shuffled_run", "clean_run"), names)
            )
            cfg = {
                "mode": "fixture",
                "analysis": {**roles, "cohorts": ["held_out"], "bootstrap": 0},
            }
            predictions = {}
            for m in names:
                file = root / (m + ".csv")
                panel[["recording_id", "route"]].assign(
                    score=rng.normal(size=len(panel))
                ).to_csv(file, index=False)
                predictions[m] = file
            analyze(cfg, panel, predictions, root / "analysis")
            write_json(root / "protocol.json", cfg)
            write_json(root / "analysis/complete.json", {"fixture": True})
            write_json(
                root / "panel/panel_audit.json",
                {"enrolled_recordings": 80, "included_recordings": 80},
            )
            report(cfg, root)
            dest = root / "paper"
            manifest = json.loads((dest / "ARTIFACTS.json").read_text())
            self.assertTrue(
                all(sha(dest / f) == h for f, h in manifest["files"].items())
            )
            self.assertEqual(
                (root / "analysis/calibration_bootstrap.csv").read_bytes(),
                (dest / "calibration_bootstrap.csv").read_bytes(),
            )
            self.assertIn("Frozen ceiling, upper", (dest / "ceiling.tex").read_text())
            self.assertIn(r"held\_out", (dest / "recording_errors.tex").read_text())
            self.assertIn(
                "Synthetic engineering fixture", (dest / "SUMMARY.md").read_text()
            )
            self.assertIn("inconclusive", (dest / "SUMMARY.md").read_text())
            figure = dest / "cohort_01_calibration.pdf"
            self.assertTrue(figure.read_bytes().startswith(b"%PDF"))
            # Rendering is deterministic, including PDF metadata and provenance receipt.
            report(cfg, root)
            self.assertEqual(
                manifest, json.loads((dest / "ARTIFACTS.json").read_text())
            )

            cfg["crop_diagnostic"] = {"enabled": True}
            crop = root / "crop"
            crop.mkdir()
            (crop / "comparison.csv").write_text("cohort,events\nheld_out,1\n")
            entry = {
                "cohort": "held_out",
                "metric": "all_tpr",
                "comparison": "tracked_minus_end_to_end:primary_gain",
                "n": 10,
                "estimate": 0.1,
                "lower": -0.05,
                "upper": 0.3,
            }
            pd.DataFrame([entry]).to_csv(crop / "paired_effects.csv", index=False)
            write_json(
                crop / "complete.json",
                {
                    "analysis": sha(root / "analysis/complete.json"),
                    "result": sha(crop / "comparison.csv"),
                    "effects": sha(crop / "paired_effects.csv"),
                },
            )
            report(cfg, root)
            self.assertIn("direction unresolved", (dest / "SUMMARY.md").read_text())
            (crop / "paired_effects.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Crop evidence"):
                report(cfg, root)
