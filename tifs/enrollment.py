"""Prespecify independent source–forgery pairs from fully linked candidate recordings."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import media
from .panel import SOURCE, read_csv, tokens, validate_sources, write_csv

LINKS = ("source_ids", "identity_ids", "session_ids")
CANDIDATE = (*SOURCE, "matched_genuine_id")
QUOTA = ("cohort", "split", "family", "pairs")


def _units(row, require_all):
    links = {("recording", row["recording_id"]), ("file", row["source_path"])}
    for field in (*LINKS, "cluster_id"):
        values = tokens(row[field])
        if field in LINKS and require_all and not values:
            raise ValueError(f"incomplete {field} for {row['recording_id']}")
        links.update((field, value) for value in values)
    return links


def _load(files, parents):
    result, paths = {}, set()
    for path in files:
        for r in read_csv(path, CANDIDATE if parents else SOURCE):
            rid = r["recording_id"]
            if not r["source_path"].strip():
                raise ValueError(f"missing source path: {rid}")
            r["source_path"] = str((path.parent / r["source_path"]).resolve())
            if rid in result or r["source_path"] in paths:
                raise ValueError(f"duplicate candidate identifier or file: {rid}")
            if r["split"] and r["split"] not in {"test", "diagnostic"} and parents:
                raise ValueError(f"candidate {rid} already assigned to {r['split']}")
            # Validate a single record. Candidate records intentionally share links across pairs.
            validate_sources(
                [
                    {
                        **r,
                        "matched_genuine_id": "",
                        "split": r["split"] or "test",
                        "cluster_id": r["cluster_id"] or rid,
                    }
                ]
            )
            result[rid] = r
            paths.add(r["source_path"])
    return result


def _plan(path):
    quotas = {}
    for row in read_csv(path, QUOTA):
        cell = (row["cohort"], row["split"], row["family"])
        n = int(row["pairs"])
        if (
            not all(cell)
            or cell[1] not in {"diagnostic", "test"}
            or n < 1
            or cell in quotas
        ):
            raise ValueError(f"invalid pair quota {cell}")
        quotas[cell] = n
    if not quotas:
        raise ValueError("empty external pair quota")
    return quotas


def _select(pools, pair_units, quotas, blocked, seed, attempt):
    # Cycle through cells. Scarce cells get first choice each round; ordering
    # within each cell is random without replacement, reproducible under row reordering.
    queues = {
        cell: iter(sorted(keys, key=lambda rid: media.key([seed, attempt, cell, rid])))
        for cell, keys in pools.items()
    }
    order = sorted(queues, key=lambda cell: (len(pools[cell]) / quotas[cell], cell))
    used, enrolled, counts = set(blocked), {}, Counter()
    while order:
        active = []
        for cell in order:
            for fake in queues[cell]:
                if used.isdisjoint(pair_units[fake]):
                    enrolled[fake] = cell
                    used.update(pair_units[fake])
                    counts[cell] += 1
                    if counts[cell] < quotas[cell]:
                        active.append(cell)
                    break
        order = active
    score = (min(counts[c] / n for c, n in quotas.items()), len(enrolled))
    return score, enrolled


def pack(inputs, quota, blocked_inputs, seed, attempts, directory):
    """Metadata-only disjoint set packing. All original IDs and discarded candidate reasons are retained."""
    directory = Path(directory)
    # A failed or changed enrollment should never leave an earlier sources.csv at this path.
    directory.mkdir(parents=True, exist_ok=False)
    if not 1 <= attempts <= 128:
        raise ValueError("attempts must be between 1 and 128")
    candidates = _load(inputs, parents=True)
    blocked = set()
    for row in _load(blocked_inputs, parents=False).values():
        blocked |= _units(row, require_all=False)
    units = {rid: _units(r, require_all=True) for rid, r in candidates.items()}
    quotas = _plan(quota)
    pools = {cell: [] for cell in quotas}
    pair_units, roles = {}, {}
    for rid, fake in candidates.items():
        target = fake["matched_genuine_id"]
        if fake["label"] == "0":
            if target:
                raise ValueError(f"unexpected genuine parent pointer for {rid}")
            continue
        real = candidates.get(target)
        if not real or real["label"] != "0" or real["cohort"] != fake["cohort"]:
            raise ValueError(f"{rid} lacks its cohort-matched authentic original")
        if any(not set(tokens(fake[k])) & set(tokens(real[k])) for k in LINKS):
            raise ValueError(
                f"{rid} has no documented parent/identity/session match to {target}"
            )
        fixed = {r["split"] for r in (real, fake) if r["split"]}
        pair_units[rid] = units[rid] | units[target]
        roles[rid] = [
            cell
            for cell in quotas
            if (cell[0], cell[2]) == (fake["cohort"], fake["family"])
            and (not fixed or fixed == {cell[1]})
        ]
        if blocked.isdisjoint(pair_units[rid]):
            for cell in roles[rid]:
                pools[cell].append(rid)

    best, enrolled, attempt_chosen = (-1, -1), {}, None
    for attempt in range(attempts):
        score, chosen = _select(pools, pair_units, quotas, blocked, seed, attempt)
        if score > best:
            best, enrolled, attempt_chosen = score, chosen, attempt
        if best[0] == 1:
            break
    accepted, ledger, occupied = {}, [], {}
    for rid, (_, split, _) in enrolled.items():
        parent = candidates[rid]["matched_genuine_id"]
        group = "pair-" + media.key([rid, parent])[:24]
        for key in (rid, parent):
            accepted[key] = {**candidates[key], "split": split, "cluster_id": group}
        occupied.update(dict.fromkeys(pair_units[rid], group))
    for rid, row in sorted(candidates.items()):
        unit = pair_units.get(rid, units[rid])
        conflicts = sorted({occupied[t] for t in unit if t in occupied})
        if rid in accepted:
            reason = "selected"
        elif unit & blocked:
            reason = "overlaps reserved material"
        elif rid in pair_units and not roles[rid]:
            reason = "family or fixed split outside registered quota"
        elif conflicts:
            reason = "overlaps another selected pair"
        elif row["label"] == "0":
            reason = "no selected forgery uses this parent"
        else:
            reason = "quota filled"
        ledger.append(
            {
                **row,
                "reason": reason,
                "selected_split": accepted.get(rid, {}).get("split", ""),
                "selected_cluster": accepted.get(rid, {}).get("cluster_id", ""),
                "conflicting_selected_clusters": "|".join(conflicts)
                if rid not in accepted
                else "",
            }
        )
    counts = Counter(enrolled.values())
    write_csv(
        directory / "candidate_ledger.csv",
        ledger,
        (
            *CANDIDATE,
            "reason",
            "selected_split",
            "selected_cluster",
            "conflicting_selected_clusters",
        ),
    )
    summary = [
        dict(zip(QUOTA, (*cell, n)))
        | {"selected": counts[cell], "shortfall": n - counts[cell]}
        for cell, n in sorted(quotas.items())
    ]
    write_csv(
        directory / "quota_coverage.csv", summary, (*QUOTA, "selected", "shortfall")
    )
    media.atomic_json(
        directory / "selection.json",
        {
            "seed": seed,
            "maximum_attempts": attempts,
            "chosen_attempt": attempt_chosen,
            "rule": "maximize minimum quota fraction, then independent pairs; earliest seeded attempt breaks ties",
            "candidate_inventory_sha256": {
                str(p.resolve()): media.digest(p) for p in inputs
            },
            "reserved_inventory_sha256": {
                str(p.resolve()): media.digest(p) for p in blocked_inputs
            },
            "quota_sha256": media.digest(quota),
        },
    )
    if best[0] < 1:
        raise ValueError(
            "unfilled independent pair quotas; inspect quota_coverage.csv and candidate_ledger.csv"
        )
    registered = [accepted[k] for k in sorted(accepted)]
    validate_sources(registered, ("diagnostic", "test"))
    write_csv(directory / "sources.csv", registered, CANDIDATE)
    return registered
