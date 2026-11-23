"""Recording-balanced loading, true paired views, and matched negative controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_frames(
    path: str, count: int | None = None, rng: np.random.Generator | None = None
):
    video = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    if (
        video.ndim != 4
        or video.shape[-1] != 3
        or video.dtype != np.uint8
        or min(video.shape) < 1
    ):
        raise ValueError(f"Invalid RGB uint8 frame cache: {path}")
    if count is None:
        array = np.array(video, copy=True)
    else:
        if not 1 <= count <= len(video):
            raise ValueError("train_frames must be within the size of each frame cache")
        indices = np.sort(rng.choice(len(video), count, replace=False))
        # Advanced indexing has already copied the requested frames from the mmap.
        array = video[indices]
    return torch.from_numpy(array).permute(0, 3, 1, 2)


def _shuffle_groups(meta, strata, fields, routes):
    """Refine each mandatory stratum on native metadata when BOTH halves support derangement."""
    if set(fields) - {"quality", "duration"}:
        raise ValueError("shuffle_quantiles accepts native quality and duration")
    grouped = list(meta.groupby(strata, sort=True, dropna=False).indices.values())
    for field in fields:
        values = meta[field].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite native shuffle feature: {field}")
        refined = []
        for locs in grouped:
            mask = values[locs] <= np.median(values[locs])
            halves = [locs[mask], locs[~mask]]
            if all(
                len(h)
                and len(h) >= routes * meta.cluster_id.iloc[h].value_counts().max()
                for h in halves
            ):
                refined.extend(halves)
            else:
                refined.append(locs)
        grouped = refined
    return grouped


class RecordingEpoch(Dataset):
    """Same unique recordings/views and update batches across every matched arm."""

    def __init__(
        self, panel: pd.DataFrame, settings: dict, objective: str, seed: int, epoch: int
    ):
        rows = panel[panel.split == "train"].copy()
        routes = list(settings["train_routes"])
        if not routes or routes[0] != "native" or len(set(routes)) != len(routes):
            raise ValueError(
                "training.train_routes must start with native and contain unique routes"
            )
        if objective != "clean" and len(routes) < 2:
            raise ValueError(
                "Paired training requires at least two training renditions"
            )
        meta = rows[rows.route == "native"].set_index("recording_id").sort_index()
        matrix = rows.pivot(
            index="recording_id", columns="route", values="frames_path"
        ).reindex(columns=routes)
        if not matrix.index.equals(meta.index) or matrix.isna().any().any():
            raise ValueError(
                "Every training recording needs every registered training route"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 101]))
        by_label = [
            rng.permutation(np.where(meta.label.to_numpy() == label)[0])
            for label in (0, 1)
        ]
        count = min(map(len, by_label))
        batch_size = int(settings["batch_size"])
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced training requires an even batch_size >= 2")
        if "steps_per_epoch" in settings:
            steps = int(settings["steps_per_epoch"])
            if steps < 1 or steps * batch_size // 2 > count:
                raise ValueError(
                    "Requested steps exceed unique balanced recordings per epoch"
                )
            count = steps * batch_size // 2
        if not count:
            raise ValueError("Training requires both labels")
        # Each consecutive pair contains exactly one real and one fake. The majority
        # class is resampled across epochs. Each recording is used once per epoch.
        chosen = np.stack([indices[:count] for indices in by_label], axis=1).ravel()
        meta = meta.iloc[chosen]
        selected_routes = routes[:1] if objective == "clean" else routes
        self.paths = matrix.loc[meta.index, selected_routes].to_numpy(dtype=object)
        self.labels = meta.label.to_numpy(dtype=np.int64)
        self.link = np.tile(np.arange(len(meta))[:, None], (1, len(selected_routes)))
        self.shuffle_diagnostics = {}
        if objective == "shuffled":
            strata = list(
                dict.fromkeys(
                    ["label", "cohort", "family", *settings.get("shuffle_strata", [])]
                )
            )
            meta = meta.copy()
            for name, bounds in settings.get("shuffle_bins", {}).items():
                if not np.all(np.isfinite(meta[name])) or not np.all(
                    np.diff(bounds) > 0
                ):
                    raise ValueError(f"Invalid shuffle stratification {name}")
                field = f"binned_{name}"
                meta[field] = np.digitize(meta[name], bounds)
                strata.append(field)
            if meta[strata].isna().any().any():
                raise ValueError("Missing shuffle stratum")
            shuffle_rng = np.random.default_rng(
                np.random.SeedSequence([seed, epoch, 202])
            )
            n_routes = self.link.shape[1]
            fields = settings.get("shuffle_quantiles", [])
            groups_by_metadata = _shuffle_groups(meta, strata, fields, n_routes)
            self.shuffle_diagnostics = {
                "mandatory_groups": meta.groupby(strata).ngroups,
                "refined_groups": len(groups_by_metadata),
                "smallest_group": min(map(len, groups_by_metadata)),
                "native_features": fields,
            }
            for locs in groups_by_metadata:
                groups = {}
                for i in locs:
                    groups.setdefault(meta.cluster_id.iloc[i], []).append(i)
                largest = max(map(len, groups.values()))
                slack = len(locs) - n_routes * largest
                if slack < 0:
                    group = meta.iloc[locs[0]][strata].to_dict()
                    raise ValueError(
                        f"Cannot derange {n_routes} routes cross-cluster in training stratum {group!r}"
                    )
                # Contiguous source blocks plus circular offsets separated by at
                # least the longest block make ALL pairs of routes independent
                # of the actual source match, including non-native / non-native.
                ordered = np.concatenate(
                    [
                        shuffle_rng.permutation(groups[k])
                        for k in shuffle_rng.permutation(list(groups))
                    ]
                )
                gaps = largest + shuffle_rng.multinomial(
                    slack, np.full(n_routes, 1 / n_routes)
                )
                offsets = shuffle_rng.permutation(np.cumsum(gaps)[:-1])
                for j, offset in enumerate(offsets, 1):
                    self.link[ordered, j] = np.roll(ordered, offset)
            for field in fields:
                values = meta[field].to_numpy(dtype=float)
                gaps = np.abs(values[:, None] - values[self.link[:, 1:]])
                self.shuffle_diagnostics[f"mean_absolute_{field}_difference"] = float(
                    gaps.mean()
                )
        self.frame_count = settings.get("train_frames")
        if self.frame_count is not None and int(self.frame_count) < 1:
            raise ValueError("train_frames must be positive")
        self.seed = seed
        self.epoch = epoch
        self.recording_ids = meta.index.to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        clips = []
        for route, parent in enumerate(self.link[index]):
            path = self.paths[parent, route]
            # Each parent supplies the exact same timestamps across matched and
            # shuffled arms, including after the association itself is changed.
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, int(parent), 303])
            )
            clips.append(read_frames(path, self.frame_count, rng))
        return torch.stack(clips), int(self.labels[index])


class SingleCopyDataset(Dataset):
    def __init__(self, panel):
        self.paths = panel.frames_path.tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return read_frames(self.paths[index])
