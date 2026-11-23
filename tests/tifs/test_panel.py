from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from tifs import importers, media, panel


def row(i, label="0"):
    return {
        "recording_id": f"r{i}",
        "label": label,
        "cohort": "study",
        "family": "real" if label == "0" else "swap",
        "cluster_id": f"c{i}",
        "split": "train",
        "source_path": f"{i}.mp4",
        "source_ids": f"source:{i}",
        "identity_ids": f"person:{i}",
        "session_ids": f"s:{i}",
    }


class Lineages(unittest.TestCase):
    def test_shared_identity_and_recording(self):
        a, b = row(0), row(1, "1")
        b["identity_ids"] += "|person:0"
        with self.assertRaisesRegex(ValueError, "shared identity"):
            panel.validate_sources([a, b])
        b["cluster_id"] = a["cluster_id"]
        panel.validate_sources([a, b])
        b["split"] = "cal"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            panel.validate_sources([a, b])

    def test_components_transitive(self):
        a, b, c = row(0), row(1), row(2)
        b["source_ids"] += "|source:0"
        c["session_ids"] += "|s:1"
        assigned, _ = importers.components([a, b, c], weights={"cal": 0.5, "dev": 0.5})
        self.assertEqual(len({r["cluster_id"] for r in assigned}), 1)
        c["split"] = "test"
        with self.assertRaisesRegex(ValueError, "crosses registered"):
            importers.components([a, b, c])

    def test_ffpp_parent_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ids = [["000", "001"], ["002", "003"], ["004", "005"]]
            cm = {}
            for split, pair in zip(("train", "val", "test"), ids):
                (root / (split + ".json")).write_text(json.dumps([pair]))
                for a, b in (pair, pair[::-1]):
                    cm[a] = f"youtube_{a} 0"
                    for sub, name in (
                        ("original_sequences/youtube", a),
                        ("manipulated_sequences/Deepfakes", a + "_" + b),
                    ):
                        f = root / sub / "c23" / "videos" / (name + ".mp4")
                        f.parent.mkdir(parents=True, exist_ok=True)
                        f.touch()
            cm["002"] = (
                "youtube_001 1"  # two extracts of the same upload across official splits
            )
            (root / "map.json").write_text(json.dumps(cm))
            rows = importers.ffpp(root, root, root / "map.json", "c23", ["Deepfakes"])
            with self.assertRaisesRegex(ValueError, "crosses registered splits"):
                importers.components(rows)
            kept, excluded = importers.components(rows, conflicts="exclude")
            self.assertEqual((len(kept), len(excluded)), (4, 8))

    def test_class_confounded_capture_session(self):
        base = {
            "kind": "measured",
            "cohort": "x",
            "split": "test",
            "route": "wire",
            "capture_method": "network_capture",
            "capture_channel": "lan",
            "capture_session": "session_a",
            "capture_settings": "h264_crf26",
            "recording_id": "g",
            "label": "0",
        }
        fake = {
            **base,
            "label": "1",
            "recording_id": "f",
            "capture_session": "session_b",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "class-confounded"):
                panel.delivery_coverage([base, fake], root, confirmatory=True)
            fake["capture_session"] = "session_a"
            self.assertEqual(
                panel.delivery_coverage([base, fake], root, confirmatory=True), 1
            )
            summary = panel.read_csv(
                root / "delivery_coverage.csv", ("genuine", "fake", "class_confounded")
            )
            self.assertEqual(
                (
                    summary[0]["genuine"],
                    summary[0]["fake"],
                    summary[0]["class_confounded"],
                ),
                ("1", "1", "0"),
            )

    def test_heldout_route_cannot_alias_known_acquisition_profile(self):
        rows = [
            {
                "kind": "measured",
                "cohort": "external",
                "split": "test",
                "route": route,
                "capture_method": "network_capture",
                "capture_channel": "h264",
                "capture_session": "capture1",
                "capture_settings": "crf=26",
                "recording_id": f"r{y}",
                "label": str(y),
            }
            for y in (0, 1)
            for route in ("known", "supposedly_new")
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "share an acquisition profile"),
        ):
            panel.delivery_coverage(rows, Path(tmp), confirmatory=True)

    def test_common_route_cannot_change_capture_profile_between_populations(self):
        rows = [
            {
                "kind": "measured",
                "cohort": cohort,
                "split": split,
                "route": "known",
                "capture_method": "network_capture",
                "capture_channel": "lab",
                "capture_session": cohort,
                "capture_settings": setting,
                "recording_id": f"{split}-{y}",
                "label": str(y),
            }
            for y in (0, 1)
            for cohort, split, setting in (
                ("source", "cal", "crf26"),
                ("external", "test", "crf21"),
            )
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(
                ValueError, "changed its registered acquisition profile"
            ),
        ):
            panel.delivery_coverage(rows, Path(tmp), confirmatory=True)

    def test_false_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            path.write_text(
                json.dumps(
                    {"recording_id": "r", "route": "foo", "source_sha256": "wrong"}
                )
            )
            with self.assertRaisesRegex(ValueError, "receipt identity"):
                panel._receipt(path, "r", "foo", "correct", "out")


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required"
)
class VideoPanel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = [row(0), row(1, "1")]
        for i, color in enumerate(("red", "blue")):
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c={color}:s=64x48:r=15:d=1",
                    "-threads",
                    "1",
                    "-c:v",
                    "libx264",
                    str(self.root / f"{i}.mp4"),
                ],
                check=True,
            )
        panel.write_csv(self.root / "s.csv", self.rows, panel.SOURCE)
        self.cfg = {
            "data": {
                "sources": str(self.root / "s.csv"),
                "frames": 4,
                "size": 32,
                "prepare_workers": 2,
                "routes": {
                    "native": {"kind": "native"},
                    "h264": {"kind": "encode", "crf": 30},
                },
            }
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_integrity_and_receipt(self):
        p = panel.prepare(self.cfg, self.root / "out")
        data = panel.read_csv(p, panel.FIELDS)
        self.assertEqual(len(data), 4)
        self.assertTrue(
            all(np.load(r["frames_path"]).shape == (4, 32, 32, 3) for r in data)
        )
        one = Path(data[0]["frames_path"])
        mtime = one.stat().st_mtime_ns
        panel.prepare(self.cfg, self.root / "out")
        self.assertEqual(one.stat().st_mtime_ns, mtime)
        one.write_bytes(b"truncated")
        panel.prepare(self.cfg, self.root / "out")
        self.assertEqual(media.digest(one), data[0]["frame_sha256"])
        cfg = self.cfg["data"]
        cfg["routes"]["receiver"] = {"kind": "measured"}
        with self.assertRaisesRegex(ValueError, "incomplete panel"):
            panel.prepare(self.cfg, self.root / "failed")
        self.assertTrue((self.root / "failed" / "attrition.csv").is_file())
        self.assertFalse((self.root / "failed" / "panel.csv").exists())

    def test_presentation_time_alignment_across_frame_rates(self):
        import numpy as np

        # Each input frame encodes its presentation time in hundredths of a second.
        for hz in (10, 20):
            path = self.root / f"timestamps-{hz}.mkv"
            frames = np.stack(
                [
                    np.full((20, 20, 3), i * 100 // hz, dtype=np.uint8)
                    for i in range(2 * hz)
                ]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "20x20",
                    "-framerate",
                    str(hz),
                    "-i",
                    "pipe:0",
                    "-c:v",
                    "ffv1",
                    str(path),
                ],
                input=frames.tobytes(),
                check=True,
                capture_output=True,
            )
            for offset, expected in [(0.07, [30, 80]), (0.15, [40, 90])]:
                # An offset between frame timestamps and an exact bin boundary exercise distinct cases.
                info = media.frames(
                    path,
                    media.digest(path),
                    1.0,
                    offset,
                    {"size": 16, "frames": 2},
                    None,
                    self.root / "cache",
                )
                self.assertEqual(
                    np.load(info["frames_path"])[:, 0, 0, 0].tolist(), expected
                )

    def test_modified_material_detected_between_jobs(self):
        import os

        import pandas as pd

        file = panel.prepare(self.cfg, self.root / "out")
        df = pd.read_csv(file)
        panel.verify_material(df)
        path = Path(df.iloc[0].frames_path)
        os.utime(path, None)  # A restaged file with identical bytes is valid.
        panel.verify_material(df, full_hash=True)
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            b = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([b[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "material no longer matches"):
            panel.verify_material(df)

    def test_content_copy_cannot_cross_groups(self):
        shutil.copyfile(self.root / "0.mp4", self.root / "1.mp4")
        with self.assertRaisesRegex(ValueError, "duplicate sha256"):
            panel.prepare(self.cfg, self.root / "out")

    def test_annotation_review_workflow(self):
        import yaml

        for r in self.rows:
            r["split"] = "test"
        panel.write_csv(self.root / "s.csv", self.rows, panel.SOURCE)
        cfg = {**self.cfg, "version": 1}
        file = self.root / "campaign.yaml"
        file.write_text(yaml.safe_dump(cfg))
        folder = self.root / "tracks"
        importers.main(
            [
                "propose-tracks",
                "--manual",
                "--config",
                str(file),
                "--out",
                str(folder),
                "--per-class",
                "1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "explicit reviewer"):
            importers.main(["finalize-tracks", "--folder", str(folder)])
        review = panel.read_csv(
            folder / "review.csv",
            (
                "recording_id",
                "route",
                "track_path",
                "media_sha256",
                "decision",
                "reviewer",
            ),
        )
        for r in review:
            r.update(decision="approved", reviewer="synthetic-test")
            target = Path(r["track_path"])
            pts = panel.read_csv(target, ("time_s", "x0", "y0", "x1", "y1"))
            for pt in pts:
                pt.update(x0="0", x1="0.7", y0="0", y1="0.8")
            panel.write_csv(target, pts, list(pts[0]))
        panel.write_csv(folder / "review.csv", review, list(review[0]))
        importers.main(["finalize-tracks", "--folder", str(folder)])
        cfg["crop_diagnostic"] = {"sources": str(folder / "sources.csv")}
        rows = panel.read_csv(
            panel.prepare_crop_diagnostic(cfg, self.root / "diagnostic"), panel.FIELDS
        )
        self.assertTrue(all(float(r["face_area_fraction"]) > 0 for r in rows))

    def test_fixed_track_subset_integrity(self):
        track = self.root / "track.csv"
        track.write_text("time_s,x0,y0,x1,y1\n0,0.05,0.15,0.8,0.9\n1,0.1,0.1,0.8,0.8\n")
        for r in self.rows:
            r["split"] = "test"
        panel.write_csv(self.root / "s.csv", self.rows, panel.SOURCE)
        subset = self.root / "subset.csv"
        fields = (*panel.SOURCE, "track_path", "track_reviewer")
        selected = [
            {**self.rows[1], "track_path": "track.csv", "track_reviewer": "fixture"}
        ]
        panel.write_csv(subset, selected, fields)
        cfg = {**self.cfg, "crop_diagnostic": {"sources": str(subset)}}
        p = panel.prepare_crop_diagnostic(cfg, self.root / "diag")
        data = panel.read_csv(p, panel.FIELDS)
        self.assertEqual(
            [r["crop_reference"] for r in data], ["fixed_track", "fixed_track"]
        )
        self.assertTrue(all(r["track_sha256"] == media.digest(track) for r in data))
        import pandas as pd

        tracked = pd.read_csv(p, keep_default_na=False)
        panel.verify_material(tracked)
        track.write_text(track.read_text().replace("0.05", "0.06"))
        with self.assertRaisesRegex(ValueError, "material no longer matches"):
            panel.verify_material(tracked)
        selected[0]["cluster_id"] = "invented"
        panel.write_csv(subset, selected, fields)
        with self.assertRaisesRegex(ValueError, "differs from original registration"):
            panel.prepare_crop_diagnostic(cfg, self.root / "diag2")

    def test_wrong_pair_with_internally_consistent_receipt(self):
        cfg = self.cfg["data"]
        captures = []
        # Construct a manifested blue clip with a different bitstream for a red recording.
        different = media.encode(
            self.root / "1.mp4",
            media.digest(self.root / "1.mp4"),
            {"kind": "encode", "crf": 40},
            self.root / "e",
        )
        records = [self.rows[0]]
        panel.write_csv(self.root / "s.csv", records, panel.SOURCE)
        captures.append(
            {
                "recording_id": "r0",
                "route": "measured",
                "path": str(different),
                "session": "one",
                "method": "archive",
                "acquired_utc": "2026-01-01T00:00:00Z",
                "alignment": "human index",
                "offset_seconds": "0",
                "channel": "archive example",
                "archive_locator": "fixture",
            }
        )
        capture = self.root / "acq.csv"
        panel.write_csv(capture, captures, list(captures[0]))
        rend = self.root / "r.csv"
        importers.main(
            [
                "register",
                "--sources",
                cfg["sources"],
                "--acquisitions",
                str(capture),
                "--receipts",
                str(self.root / "rc"),
                "--out",
                str(rend),
            ]
        )
        cfg.update(
            renditions=str(rend),
            routes={"native": {"kind": "native"}, "measured": {"kind": "measured"}},
        )
        with self.assertRaisesRegex(ValueError, "alignment candidates"):
            panel.prepare(self.cfg, self.root / "out")

    def test_network_receipt_smoke(self):
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        cfg = self.cfg["data"]
        args = [
            "--sources",
            cfg["sources"],
            "--route",
            "wire",
            "--session",
            "smoke",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--splits",
            "train",
            "--timeout",
            "20",
        ]
        receiver = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tifs.media",
                "receive",
                *args,
                "--out",
                str(self.root / "received"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(0.6)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tifs.media",
                    "send",
                    *args,
                    "--out",
                    str(self.root / "tx"),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            _, err = receiver.communicate(timeout=15)
            self.assertEqual(receiver.returncode, 0, err)
        finally:
            if receiver.poll() is None:
                receiver.kill()
                receiver.communicate()
        cfg["renditions"] = str(self.root / "received" / "renditions.csv")
        cfg["routes"] = {"native": {"kind": "native"}, "wire": {"kind": "measured"}}
        result = panel.read_csv(
            panel.prepare(self.cfg, self.root / "out"), panel.FIELDS
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(
            all(len(r["receipt_sha256"]) == 64 for r in result if r["route"] == "wire")
        )
        rec = panel.read_csv(Path(cfg["renditions"]), panel.MEASURED)
        receipt = Path(rec[0]["receipt_path"])
        note = json.loads(receipt.read_text())
        note["recording_id"] = "wrong"
        receipt.write_text(json.dumps(note))
        import pandas as pd

        registered = pd.read_csv(self.root / "out" / "panel.csv", keep_default_na=False)
        with self.assertRaisesRegex(ValueError, "material no longer matches"):
            panel.verify_material(registered)
        cfg["allow_incomplete"] = True
        with self.assertRaisesRegex(panel.ProvenanceError, "receipt identity"):
            panel.prepare(self.cfg, self.root / "tampered")


if __name__ == "__main__":
    unittest.main()
