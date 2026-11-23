# Optional FF++ Social check

Marcon, Pasquini and Boato distributed Facebook and YouTube renditions of FF++ validation and test recordings. Those FF++ partitions serve as source development and calibration here. This check applies the locked primary and ordinary-augmentation detectors and common thresholds to native and archived copies. It describes the enrolled source subset and cannot supply an independent external-cohort result. The delivered files reflect the historical platform pipelines. [Archive catalogue](https://mmlab.science.unitn.it/resources/published-datasets) · [Original study](https://doi.org/10.3390/jimaging7100193)

Use the same checkout, environment, mounted data and registered configuration as the completed campaign. Supply the licensed archive, its mapping to the precise FF++ input rendition used for uploading, and a reviewed acquisition ledger. Same numeric stem alone cannot verify parentage, manipulation type or input compression. Record the archival release/item locator and the visual time alignment. Keep the acquisition and auxiliary result directories outside the main output.

```bash
CFG=configs/tifs/campaign.yaml
A="$TIFS_INVENTORY/ffpp-social"
mkdir -p "$A"
python -m tifs.archive sources --config "$CFG" --out "$A/sources.csv"
```

This snapshots every enrolled FF++ dev/cal source. To preregister fewer families, include `--families real Deepfakes Face2Face FaceSwap NeuralTextures` on the same command with a new output file. Record any other enrollment restriction before inspecting scores. The number of files in the full published archive is distinct from the number of independent source lineages here.

Provide `$A/acquisitions.csv` with one reviewed mapping per enrolled recording and registered social route. Its columns are `recording_id,route,path,session,method,channel,settings,acquired_utc,alignment,offset_seconds,archive_locator`. Accepted archive rows use `method=archive`. Use the release-specific stored file as `path`, a stable archive item identifier as `archive_locator`, a timezone-aware actual acquisition timestamp, and registered route names such as `social_facebook` and `social_youtube`. Document the published processing or settings actually known. Never infer upload input provenance from a thumbnail or substitute a local transcode. The general registration tool creates byte-bound receipts. The auxiliary checker also verifies every native recording, parent and donor lineage field, input hash, frame-stack hash and analysis window against the primary panel.

```bash
python -m tifs.importers register --sources "$A/sources.csv" \
  --acquisitions "$A/acquisitions.csv" --receipts "$A/receipts" --out "$A/renditions.csv"
python -m tifs.archive --config "$CFG" --sources "$A/sources.csv" \
  --renditions "$A/renditions.csv" --routes social_facebook social_youtube \
  --out "${TIFS_OUTPUT}-ffpp-social"
```

Reuse the same command to resume. Native predictions come from the sealed campaign. Only new archive copies require inference. `metrics.csv` reports route, all-copy and any-copy rates pooled and by original source split. `coverage.csv` reports recording and independent-unit counts. A strict complete-panel check stops on missing enrolled media. `registration.json`, `panel/`, per-run `all_scores.csv` and `complete.json` retain settings, verified weights, cutoffs, receipts and artifact hashes. Pointwise Clopper–Pearson intervals on one deterministic representative per class and lineage summarize count precision. Reuse of these source cases in development or calibration limits inference from those intervals.
