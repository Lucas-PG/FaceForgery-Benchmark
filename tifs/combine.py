"""Combine independently acquired route inventories without changing relative-path meanings."""

import argparse
from pathlib import Path

from .panel import MEASURED, read_csv, write_csv


def combine(inputs, output):
    rows = []
    keys = set()
    sources = [Path(p).resolve(strict=True) for p in inputs]
    if Path(output).resolve() in sources:
        raise ValueError("Use a separate output inventory")
    for source in sources:
        for entry in read_csv(source, MEASURED):
            key = entry["recording_id"], entry["route"]
            if key in keys:
                raise ValueError(f"Duplicate measured recording and route: {key}")
            keys.add(key)
            for field in ("path", "receipt_path", "track_path"):
                if entry.get(field):
                    entry[field] = str(
                        (source.parent / entry[field]).resolve(strict=True)
                    )
            rows.append(entry)
    if not rows:
        raise ValueError("No acquired renditions")
    fields = list(MEASURED) + sorted(
        set().union(*(r.keys() for r in rows)) - set(MEASURED)
    )
    write_csv(
        Path(output),
        sorted(rows, key=lambda r: (r["recording_id"], r["route"])),
        fields,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    combine(args.inputs, args.out)
