"""Fail early on incomparable arms, broken holdouts and unavailable finite-sample designs."""

from pathlib import Path


def check_matrix(cfg, jobs):
    roles = cfg["analysis"]
    required = [*roles.get("control_runs", []), *roles.get("selection_runs", [])]
    if set(required) - jobs.keys():
        raise ValueError(
            "Analysis references unscheduled controls or selection candidates"
        )
    if cfg.get("mode") not in {"fixture", "confirmatory", "exploratory"}:
        raise ValueError("Set mode to confirmatory, exploratory or fixture")

    def comparison_settings(spec):
        return {
            k: v for k, v in spec.items() if k not in {"objective", "name", "seeds"}
        }

    matched = {}
    for spec in cfg["models"]:
        if spec["objective"] == "frozen":
            continue
        arm = (comparison_settings(spec), list(spec.get("seeds", cfg["seeds"])))
        if matched.setdefault(spec["backbone"], arm) != arm:
            raise ValueError(
                f"Confounded settings, assets or training seeds for {spec['name']}"
            )
    for role in [roles, *roles.get("replications", [])]:
        refs = [
            role[k]
            for k in ("primary_run", "baseline_run", "shuffled_run", "clean_run")
        ]
        all_refs = refs + (roles.get("control_runs", []) if role is roles else [])
        anchor, seed = jobs[refs[0]]

        for name in all_refs:
            spec, runseed = jobs[name]
            if runseed != seed or comparison_settings(spec) != comparison_settings(
                anchor
            ):
                raise ValueError(
                    f"Confounded comparison: {name} differs in data, model, assets, seed or settings"
                )
        expected = ["constrained", "augmented", "shuffled", "clean"]
        if [jobs[k][0]["objective"] for k in refs] != expected:
            raise ValueError(
                "Mismatched approved objectives in primary/replication roles"
            )
    if cfg["mode"] == "confirmatory":
        if "tiny" in {s["backbone"] for s, _ in jobs.values()}:
            raise ValueError("Fixture backbones cannot support confirmatory evidence")
        confirm = roles.get("confirmation_cohorts", roles["cohorts"])
        if (
            len(set(confirm)) < 2
            or not set(confirm) <= set(roles["cohorts"])
            or len(set(confirm)) != len(confirm)
        ):
            raise ValueError(
                "Confirmatory protocol requires two independent target cohorts"
            )
        if cfg["data"].get("allow_incomplete", False):
            raise ValueError(
                "Incomplete source panels require a separate exploratory study"
            )
        if cfg["data"].get("crop") == "full":
            raise ValueError(
                "Release uses a registered face crop policy; full-frame fallback must be counted"
            )


def check_panel(cfg, panel, directory):
    import numpy as np

    from .stats_core import representatives, split_dev, wide
    from .stats_power import power_audit
    from .training_data import RecordingEpoch

    if set(panel.split) != {"train", "dev", "cal", "diagnostic", "test"}:
        raise ValueError("Need train/dev/cal/diagnostic/test partitions")
    source = panel[panel.split.isin(["train", "dev", "cal"])]
    target = panel[panel.split.isin(["diagnostic", "test"])]
    expected = set(cfg["analysis"]["cohorts"])
    if set(target.cohort) != expected or set(source.cohort) & expected:
        raise ValueError("Source and target recording cohorts must be disjoint")
    train_routes = cfg["training"]["train_routes"]
    if train_routes[0] != "native" or len(train_routes) != len(set(train_routes)):
        raise ValueError("Declare distinct training routes with native first")
    actual = {
        s: set(source.loc[source.split == s, "route"]) for s in ("train", "dev", "cal")
    }
    if actual["train"] != set(train_routes):
        raise ValueError(
            "Source training routes differ from registered objective inputs"
        )
    if actual["dev"] != actual["cal"] or not actual["train"] <= actual["dev"]:
        raise ValueError(
            "Source development/calibration must share routes and cover all training routes"
        )
    for c in expected:
        subsets = [
            target[(target.cohort == c) & (target.split == s)]
            for s in ("diagnostic", "test")
        ]
        if any(x.empty for x in subsets) or set(subsets[0].route) != set(
            subsets[1].route
        ):
            raise ValueError(f"Missing or different diagnostic/test routes in {c}")
        if set(subsets[1].label) != {0, 1} or 0 not in set(subsets[0].label):
            raise ValueError(
                f"Missing genuine diagnostic or sealed-test classes in {c}"
            )
        if cfg["mode"] == "confirmatory":
            measured = set(subsets[1].loc[subsets[1].kind == "measured", "route"])
            if (
                len(measured) < 2
                or not measured - actual["dev"]
                or not measured & actual["dev"]
            ):
                raise ValueError(
                    f"{c}: need known and independently held-out measured delivery routes"
                )
    if cfg["mode"] == "confirmatory":
        fake = target[(target.label == 1) & (target.split == "test")]
        declared = cfg["analysis"].get("heldout_families", {})
        if not declared:
            raise ValueError(
                "Declare the independently verified held-out family in its target cohort"
            )
        trained = set(
            source.loc[source.label == 1, "family"].str.strip().str.casefold()
        )
        for cohort, families in declared.items():
            observed = set(
                fake.loc[fake.cohort == cohort, "family"].str.strip().str.casefold()
            )
            requested = {x.strip().casefold() for x in families}
            if not requested or requested & trained or not requested <= observed:
                raise ValueError(
                    f"Unverified or missing prespecified held-out family: {cohort}/{families}"
                )
        if (
            not source.identity_ids.astype(str).str.strip().ne("").all()
            or not target.identity_ids.astype(str).str.strip().ne("").all()
        ):
            raise ValueError(
                "Supply audited identity identifiers for the confirmatory design"
            )
    dev = source[source.split == "dev"].assign(score=0.0)
    split_dev(representatives(wide(dev)))
    if not np.isfinite(panel.quality.to_numpy(float)).all():
        raise ValueError("Nonfinite received-image quality feature")
    power = power_audit(cfg, panel)
    power.to_csv(directory / "power.csv", index=False)
    print(power.to_string(index=False), flush=True)
    confirm = cfg["analysis"].get("confirmation_cohorts", cfg["analysis"]["cohorts"])
    mandatory = power[power.cohort.isin(confirm)]
    if cfg["mode"] == "confirmatory" and (
        (mandatory.maximum_passing_false_alerts < 0).any()
        or (mandatory.diagnostic_quantile_rank < 1).any()
    ):
        raise ValueError(
            "Registered cohort cannot resolve the false-alarm/ceiling design even with zero observed errors; inspect power.csv"
        )
    for spec in cfg["models"]:
        if spec["objective"] == "shuffled":
            settings = {**cfg["training"], **spec.get("training", {})}
            for seed in spec.get("seeds", cfg["seeds"]):
                for epoch in range(int(settings["epochs"])):
                    RecordingEpoch(
                        source[source.split == "train"],
                        settings,
                        "shuffled",
                        seed,
                        epoch,
                    )
    return power


def inventory_power(cfg, directory):
    """Score-independent design screen before receivers, encoders or GPUs are needed."""
    import pandas as pd

    from .panel import SOURCE, read_csv, validate_sources
    from .stats_power import power_audit

    d = cfg["data"]
    rows = read_csv(Path(d["sources"]), SOURCE)
    validate_sources(rows, d.get("require_identity_splits", []))
    table = pd.DataFrame(rows)
    table["route"] = table.split.map(
        lambda s: list(d.get("routes_by_split", {}).get(s, d["routes"]))
    )
    power = power_audit(cfg, table.explode("route"))
    directory.mkdir(parents=True, exist_ok=True)
    power.to_csv(directory / "enrollment_power.csv", index=False)
    print(power.to_string(index=False), flush=True)
    confirm = cfg["analysis"].get("confirmation_cohorts", cfg["analysis"]["cohorts"])
    mandatory = power[power.cohort.isin(confirm)]
    if cfg["mode"] == "confirmatory" and (
        (mandatory.maximum_passing_false_alerts < 0).any()
        or (mandatory.diagnostic_quantile_rank < 1).any()
    ):
        raise ValueError(
            "Enrolled independent lineages cannot resolve the registered gates. See enrollment_power.csv"
        )
    return power
