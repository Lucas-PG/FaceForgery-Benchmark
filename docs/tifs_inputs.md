# Preparing and registering video evidence

Run from the repository root in the campaign environment. FFmpeg and ffprobe with libx264 are required. Keep media, faces, raw inventories, measurements and linkage keys on approved storage. Filesystem paths in CSVs resolve relative to that CSV. The recording inventory defines the study population **before** examining scores. Register all intended cases, including known failed acquisitions. Set `TIFS_INVENTORY` to the registered input directory used by the campaign YAML.

## Source inventory

`sources.csv` has these columns. The optional `track_path` is an acquisition-matched normalized face track.

| Column | Contract |
|---|---|
| `recording_id, cohort, family, label` | Unique persistent identifier for the underlying labeled recording, independent-capture cohort, actual visual manipulation family and `0` genuine or `1` forged. For genuine media use `real`. Exclude audio-only fakes from the preregistered population. |
| `source_ids, identity_ids, session_ids` | Pipe-separated global keys for all known parent footage, subjects (donor **and** target), and original acquisition sessions. Namespace keys so the same person/source shared across releases resolves to the same key. Never substitute unique recording IDs for missing actor identities. Source IDs are required. Blank identity/session fields mean undocumented and are reported; `require_identity_splits: [diagnostic, test]` makes missing external identity IDs an error. |
| `cluster_id, split, source_path` | Connected component for resampling/splitting, one of `train/dev/cal/diagnostic/test`, path to native video. For `inventory` input use an empty cluster when it should be derived. `diagnostic` is independent external reference footage and `test` is the locked outcome sample. |
| `track_path` | Optional CSV with strictly increasing `time_s,x0,y0,x1,y1`, 0–1 boxes in display coordinates. It must cover sampled timestamps; no unregistered extrapolation. |

**FF++.** Use the original directory structure, official split JSON and upstream `conversion_dict.json`. The importer expects every enrolled parent and both manipulation directions for each selected family, recognizes excerpts from the same YouTube upload, and keeps donor/target partners together. It maps official train/validation/test to development train/dev/cal. Those 1,000 originals do not represent 1,000 independent identities. `--identities ffpp_people.csv` can supply independently verified `original_id,identity_ids,session_ids`. Source-disjoint is the strongest available claim where actor metadata are absent. Conflicting components fail by default. `--conflicts exclude` discards the entire component and records each exclusion before model scores are opened. [Official data layout and splits](https://github.com/ondyari/FaceForensics/tree/master/dataset)

```bash
python -m tifs.importers ffpp --root "$FFPP_ROOT" --codec c23 \
  --splits "$FFPP_METADATA/splits" --conversion "$FFPP_METADATA/conversion_dict.json" \
  --identities "$TIFS_INVENTORY/ffpp_people.csv" --conflicts exclude --out "$TIFS_INVENTORY/ffpp.csv"
```

**DFDC, KoDF, other independent recordings.** Original-to-fake index links in DFDC do not document every donor or establish that a fake is visual. Use the laboratory's verified visual subset. DFDC `decisions.csv` needs `relative_path,visual_label,family,identity_ids,session_ids,split,source_ids`. `relative_path` locates a file under its official shard; the importer checks sibling `metadata.json` and adds the official original to both labels' grouping keys. KoDF or newly recorded cohorts use the common schema above. Verify class labels, generator, source/target and actor/session crosswalks from the release and any curation before importing. A permission to analyze a dataset does not itself authorize uploading a person's footage to an external service. For a random split, register its seed and proportions once at the connected-component level. Review effective class/cluster counts before inference. Hash splits alone do not guarantee balance. [DFDC](https://ai.meta.com/datasets/dfdc/) · [KoDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Kwon_KoDF_A_Large-Scale_Korean_DeepFake_Detection_Dataset_ICCV_2021_paper.pdf)

```bash
python -m tifs.importers dfdc --root "$DFDC_ROOT" --decisions "$TIFS_INVENTORY/dfdc_decisions.csv" \
  --out "$TIFS_INVENTORY/dfdc.csv"
python -m tifs.importers inventory "$TIFS_INVENTORY/kodf_reviewed.csv" \
  --weights '{"diagnostic":0.25,"test":0.75}' --seed 42 --out "$TIFS_INVENTORY/kodf.csv"
python -m tifs.importers inventory "$TIFS_INVENTORY/ffpp.csv" "$TIFS_INVENTORY/dfdc.csv" \
  "$TIFS_INVENTORY/kodf.csv" --out "$TIFS_INVENTORY/sources.csv"
```

### Independent enrollment when actor/donor networks connect many candidates

A broad candidate pool can contain thousands of fakes, yet few independent lineages. For a prospective matched study, select source–forgery pairs **before** receiving copies or generating scores. Audit all target and donor identity, physical source and recording-session IDs, and each fake's actual authentic parent. KoDF candidates use the source schema plus `matched_genuine_id`, blank on genuine rows. DFDC candidate export derives that pointer from the official `original` entry. Include its genuine row. A known original split or indivisible cluster stays binding. Leave unassigned `split` and `cluster_id` blank at candidate intake. Run this on the eligible raw pool before graph closure. Keep existing cohort partitions if already preregistered.

Create `quotas.csv` with `cohort,split,family,pairs` and one preregistered target count per external diagnostic/test × enrolled forgery family cell. Each pair contributes one genuine and one forged recording. Set counts from the precision/power design and actual documented candidate supply. Freeze the quota file, candidate inventories, seed and number of attempts before observing model outputs.

```bash
python -m tifs.importers dfdc --root "$DFDC_ROOT" --decisions "$TIFS_INVENTORY/dfdc_decisions.csv" \
  --candidate-pool --out "$TIFS_INVENTORY/dfdc_candidates.csv"
python -m tifs.importers enroll-pairs "$TIFS_INVENTORY/dfdc_candidates.csv" "$TIFS_INVENTORY/kodf_candidates.csv" \
  --quotas "$TIFS_INVENTORY/quotas.csv" --exclude-sources "$TIFS_INVENTORY/ffpp.csv" \
  --seed 42 --attempts 16 --out "$TIFS_INVENTORY/external-pairs"
python -m tifs.importers inventory "$TIFS_INVENTORY/ffpp.csv" "$TIFS_INVENTORY/external-pairs/sources.csv" \
  --out "$TIFS_INVENTORY/sources.csv"
```

The selector uniformly randomizes candidate order within each quota by seed, prioritizes scarce quota cells, and greedily enforces disjoint recording, footage, person, original session and declared cluster keys across accepted pairs. It maximizes minimum quota fulfillment then pair count across the fixed attempts. A packing shortfall can reflect the heuristic or supply; the ledger and quota report survive and the command produces no enrollment. Keep the whole candidate ledger, including rejected bridge fakes, and publish eligible-to-enrolled counts and family composition. Inferences apply to the selected matched populations. Neither a large raw video count nor removal of a connecting candidate establishes independence when a subject, source or acquisition session is missing from the enrolled metadata.

## Measured delivery

For a real platform download, archived FF++ Social or an optical recapture, maintain an acquisitions CSV with `recording_id,route,path,session,method,channel,settings,acquired_utc,alignment,offset_seconds`. Methods are `platform_download`, `optical_recapture`, `network_capture`, or `archive`. Supply `archive_locator` for archival media, naming the upstream file and release. Include the ISO-8601 timestamp with timezone, actual platform/app/device/version and settings in `channel`, and how the *matching visual content* was identified in `alignment`. Record leading padding as the positive delay from source zero to the corresponding received presentation time. Time-warped, incomplete, or edited excerpts require explicitly synchronized source/received excerpts registered before analysis. Use a shared session ID for an actual acquisition block. Put the nominal encoder/device/app/profile settings in `settings`; retain per-clip realized bitrate and other observations outside this grouping key. Keep genuine and forged clips intermixed in each tested cohort/route/session/settings combination. The confirmatory panel rejects one-label test cells and missing settings. A genuinely authentic-only diagnostic fold is permissible. Report `delivery_coverage.csv` to distinguish replication across independent sessions from repeat measurements within a single session. Capture flags are evidence from the acquisition operator, not automated proof a file traversed a commercial platform. [Paired FF++ social redistribution study](https://www.mdpi.com/2313-433x/7/10/193)

```bash
python -m tifs.importers register --sources "$TIFS_INVENTORY/sources.csv" \
  --acquisitions "$TIFS_INVENTORY/acquisitions.csv" --receipts "$TIFS_INVENTORY/receipts" \
  --out "$TIFS_INVENTORY/renditions.csv"
```

There is also a receiver-verified controlled H.264 network route. Run the receiver first and then the sender on **distinct authorized lab endpoints**. Both need the same ordered sources and registered source bytes, route, session and CRF. Select registered and permitted clips. Collect the measured `network` route for source dev/cal and for external diagnostic/test. Collect the independent `redistribution` route for external diagnostic/test only, and register that operator’s observed deliveries using the acquisitions CSV. Use a private interface or an SSH TCP forward, with the receiver bound to loopback for a tunnel. The receiver hashes delivered TS bytes and verifies an in-band tag binding the original digest, recording, session and route. Retrying both commands skips already verified captures. A loopback test validates the software only. This reliable stream provides measured network delivery with a specified encoder; commercial adaptive WebRTC, packet loss and independent platform behavior require their own receiver acquisitions. Download/recover the delivered file, rather than labeling a local re-encode as measured.

```bash
# Receiver. Use the private numeric IP here, or 127.0.0.1 if using a tunnel.
python -m tifs.media receive --sources "$TIFS_INVENTORY/sources.csv" \
  --splits dev cal diagnostic test --route network --session round1 --crf 26 \
  --host "$BIND_IP" --port 29418 --out "$CAPTURE_DIR"
# Sender. Use the receiver IP, or the sender's local SSH-forward endpoint.
python -m tifs.media send --sources "$TIFS_INVENTORY/sources.csv" \
  --splits dev cal diagnostic test --route network --session round1 --crf 26 \
  --host "$RECEIVER_IP" --port 29418 --out "$SENDER_DIR"
```

The receiver produces `renditions.csv` and one TS plus JSON receipt per recording. If multiple acquisition registers are combined, concatenate their rows under one header and keep paths absolute. Names must exactly match the YAML route keys. Each measured route/recording requires its own observation. For each named route, keep the prescribed method, channel and nominal settings fixed across source and external cohorts and across diagnostic/test. Sessions may differ. Interpret results conditionally on these supervised acquisition profiles and sampled recording populations. Two measured route labels resolving to an identical method/channel/settings tuple invalidate the distinct-route comparison. Different sessions using the same profile are replications of that route. With `routes_by_split`, predeclare different development and external channels; each cohort/split uses the same listed channels for both labels. Training, dev and cal should not have access to a sealed external delivery route.

## Pixel pipeline and coverage

`native: {kind: native}` reads the original. `{kind: encode, crf: 28, preset: medium, height: 480}` creates a local H.264 stressor. `{kind: measured}` requires the registered bytes and receipt. `frames` RGB samples use one uniform native presentation-time grid mapped to each rendition. `window_start_seconds`, optional `window_seconds` and `tail_guard_seconds` (default 0.05) fix the common interval; too-short received clips fail. Require sufficient duration for the selected temporal comparator. `size` is the square model input-cache size and `decode_max_side` caps decoding resolution, default 1024. All stages use RGB, aspect-preserving resize and lossless uint8 NPY caching. Capture actual delivered timing, avoid frame-index matching through frame-rate changes. Quality is log(1 + mean squared horizontal and vertical gray gradients) on the received cached view. It is a source-fitted baseline feature, never a route label.

For a spatial end-to-end run, `crop: detected` uses OpenCV's frontal-face cascade per sampled frame. Missing/ambiguous detections use full frame and are explicitly counted by default. `on_face_failure: error` is the strict alternative. `crop: fixed_track` uses the same source track across renditions and requires explicitly attested geometry preservation on measured routes. `crop: view_track` takes separately annotated tracks from native and received CSVs. `crop: full` uses letterboxed full frames. These are distinct experimental procedures. Rerun a diagnostic preprocessing policy in a separate output directory, using the same starts, population and splits.

`panel.csv` binds each recording/route to media digest, source digest, receipt, source and track hashes and annotation paths, frame-stack path/hash, sampled interval and full native duration, count of frames with a localized face, their median face-box/display-area fraction, and crop failures. `panel_audit.json`, `attrition.csv`, `alignment_audit.csv` and `near_duplicates.csv` are pre-model outputs. Byte-identical native enrollments, exact cross-split content, and identical received files assigned to unrelated measured cases are errors. Within-split transformed/cropped views of distinct linked genuine and forged recordings can legitimately become identical. They remain in evaluation and are recorded in `content_collisions.csv`. A three-timestamp perceptual screen uses a middle-frame difference hash (distance ≤3) and small RGB thumbnail difference (≤2 levels) to flag near duplicates across declared independent clusters. A separate low-frequency color/correlation screen flags gross mispaired versions even when a mistaken receipt has matching file hashes. These checks can miss subtly re-edited, time-shifted or recolored copies. Independently verify lineage and spot-review registration by cohort/route/class.

Correct suspected lineage errors by regrouping sources or fixing acquisition provenance, then rerun before opening scores. Documented nonduplicates may be cleared via `duplicate_clearances` CSV with `left_sha256,right_sha256,reviewer,reason`. Verified but visually altered true renditions use `alignment_clearances` CSV with `source_sha256,received_sha256,reviewer,reason`. Store these with the dataset release. An altered input invalidates its hash-bound adjudication.

Primary analysis requires a complete panel. Missing views or unreadable media fail preparation and leave per-view causes in the attrition ledger. The exploratory switch `allow_incomplete: true` removes entire incomplete recordings across all views; explicitly account for this denominator and suppress confirmatory claims whenever incomplete records could change the finding.

## Frozen-model crop diagnostic

`crop_diagnostic.sources` is a preregistered subset of the primary inventory on `diagnostic/test`, retaining exactly the same recording, family, label, lineage, native path and split. Add `track_path,track_reviewer` with the identity of the annotation reviewer. `crop_diagnostic.renditions` contains the corresponding subset of the same acquisition paths and receipts. The native track propagates to simulated encodes and to measured deliveries with `geometry_preserved=true` in the receipt. For all other measured deliveries provide independently registered normalized boxes in its own display coordinates plus `track_reviewer`. This holds the face region as comparable as the measured geometry permits. The diagnostic uses the primary duration, timestamps, size, and held-out route panel. Evaluate the frozen primary model and comparator thresholds on exactly these keys and disclose the selected coverage.

Generate the diagnostic enrollment and editable tracks **before scoring**, using a fixed seed and desired independent clusters per label/cohort/split. A fresh draft contains `selection.json`, the selected keys, and one PNG in display coordinates per sampled timestamp alongside editable normalized `boxes.csv` files. The helper proposes boxes only for unambiguous frontal detections. Use `--manual` to start from empty coordinates. For **every** registered track, inspect the PNGs, adjust all boxes to the target face, then enter the reviewer's ID and `approved` in `review.csv`. Empty coordinates, missing decisions, revised media or inconsistent geometry fail the finalization. A genuinely untrackable registered clip must be reported; silently choosing a replacement would change the estimand.

```bash
python -m tifs.importers propose-tracks --config configs/tifs/campaign.yaml --splits test \
  --per-class 40 --seed 20261123 --out "$TIFS_INVENTORY/tracked"
# The lab reviews the selected traces in the draft as described above.
python -m tifs.importers finalize-tracks --folder "$TIFS_INVENTORY/tracked"
```

Use `$TIFS_INVENTORY/tracked/sources.csv` and `$TIFS_INVENTORY/tracked/renditions.csv` in the frozen crop diagnostic configuration. The selected tracks require actual visual review; a detector proposal is not automatically approved. If the production configuration requires identity IDs in all splits, pass the verified `--identities "$TIFS_INVENTORY/ffpp_people.csv"` to the FF++ command above, and do not claim identity-disjoint inference from YouTube upload IDs.
