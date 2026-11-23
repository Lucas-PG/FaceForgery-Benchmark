# Running the paired-video campaign

Run from the branch checkout. The YAML is the registered design. Keep the exact revision, input manifests, source footage, delivery receipts, weights, and output root available to all jobs. Full execution requires the lab's legal access to FF++, DFDC, KoDF, reviewed source/donor/subject/session links, verified visual labels, and the two measured delivery pipelines. These private inputs cannot be shipped with the code.

## Environment and storage

On CISIA, create the environment from Shell Access. Run preprocessing, tests, training, and analysis under a SLURM allocation. Install FFmpeg/ffprobe with libx264 from the existing system module or conda-forge. GPU jobs use the repository's CUDA 12.8 wheel pair. Choose writable **shared** storage for all persistent paths. A node-local cache or media mount needs a single node selected with `--node` at submission.

```bash
source /opt/conda/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="$HOME/.conda/envs"
conda create -y -n tifs -c conda-forge python=3.12 pip ffmpeg
conda activate tifs
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-tifs.txt
python -m pip check
export TIFS_ENV=tifs
export TIFS_ROOT="/projects/models/$USER/tifs"
export TIFS_INVENTORY="$TIFS_ROOT/inventory" TIFS_CACHE="$TIFS_ROOT/media-cache"
export TIFS_ASSETS="$TIFS_ROOT/assets" TIFS_OUTPUT="$TIFS_ROOT/campaign-v1"
export TIFS_INPUTS="$TIFS_INVENTORY"
mkdir -p "$TIFS_INVENTORY" "$TIFS_CACHE" "$TIFS_ASSETS" "$TIFS_OUTPUT"
python -m pip freeze > "$TIFS_ASSETS/environment.txt"
ffmpeg -version > "$TIFS_ASSETS/ffmpeg.txt"
```

## Register the data and assets

1. Use the FF++, DFDC and inventory commands in [the input contract](tifs_inputs.md). Include the FF++ `--identities` crosswalk covering both partners, verified visual DFDC decisions, and KoDF FSGAN positives with audited links. Assign independent components to partitions before acquisition. An identity crosswalk or source session known to be the same across two releases must use the same key. Review the source rows and run the enrollment screen before transmitting any footage. Inspect projected power, particularly for native noninferiority. At the registered multiplicity, 428 independent forged lineages are needed for the two-point native comparison to pass when there are zero observed discordances. KoDF contains 403 participants in its original release, and shared donors further constrain independent enrollment. A third, independently enrolled cohort can be prospectively designated for confirmation while KoDF remains visible as a transport cohort. See [the power scenarios and cohort settings](tifs_statistics.md) before requesting GPU time.
2. Deliver the source dev/cal and external diagnostic/test videos over `network`. Independently acquire the external diagnostic/test `redistribution` route, e.g. a controlled optical recapture or licensed platform. Record exact timing, settings, sessions and relevant display geometry. The two routes must represent genuinely different processes; labels alone cannot establish that. Balance conditions across classes within sessions. Retain same-content inspections, receipt hashes and any documented clearances. For every test video, register native plus both measured receipts; use the precise commands in the input contract. Combine route inventories with `python -m tifs.combine "$CAPTURE_DIR/renditions.csv" "$REDISTRIBUTION_DIR/renditions.csv" --out "$TIFS_INVENTORY/renditions.csv"`. Generate the preregistered crop sample and have an annotator review its frames **before model scoring**.

```bash
CFG=configs/tifs/campaign.yaml
python -m tifs.campaign plan --config "$CFG"
python -m tifs.campaign enrollment --config "$CFG"
python -m tifs.importers propose-tracks --config "$CFG" --splits test --per-class 40 \
  --seed 20261123 --out "$TIFS_INVENTORY/tracked"
# After reviewing the selected timestamps, correcting boxes and signing review.csv:
python -m tifs.importers finalize-tracks --folder "$TIFS_INVENTORY/tracked"
```

3. Supply the initialization snapshots produced by `scripts/setup_pretrained_cisia.py` (headless DINO and the CLIP vision-only directory). Adjust `PRETRAIN` to their shared location. The asset command downloads both seed-42 historical detectors and their original configurations from the authors' pinned release. To reuse identical archived files instead, pass optional `--dino-tcc` and `--clip-tcc` paths with their sibling `results/run_config.json` present. The published configurations use the generic regime name `finetune`; their pinned weights, configurations and original campaign records identify the robust runs. Run acquisition where network access is available. Heavy backbone preparation belongs in an allocation. [Authors’ released models](https://huggingface.co/lucasoc/MFFI-Models/tree/b408d7572fcc04638ce20adee0546471d46dce2a)

```bash
PRETRAIN=/datasets/ai_models/faceforgery/pretrained
gdown 1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7 -O "$TIFS_ASSETS/lipforensics.pth"
curl -fL --retry 3 https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  -o "$TIFS_ASSETS/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -dkf "$TIFS_ASSETS/shape_predictor_68_face_landmarks.dat.bz2"
python -m tifs.assets --directory "$TIFS_ASSETS" --dino-initial "$PRETRAIN/dino/base.pth" \
  --clip-initial "$PRETRAIN/clip" \
  --lip-weights "$TIFS_ASSETS/lipforensics.pth" --landmarks "$TIFS_ASSETS/shape_predictor_68_face_landmarks.dat"
```

The asset command pins the GenD and LipForensics upstream revisions, downloads the released GenD CLIP-L/14 weights and processor, and binds asset/configuration hashes. At preflight every distinct architecture loads its weights strictly and performs a forward pass. Unsupported/missing archived weights or acquisition metadata stop the campaign. An authenticated provider download or lab archive of the identical file works if a public download service is unavailable. [GenD release](https://github.com/yermandy/GenD) · [LipForensics release](https://github.com/ahaliassos/LipForensics)

## Validate and submit

From an allocated shell, run the pipeline fixture. With a single-face video of at least one second and the registered weights, run the independent upstream parity tests as well. The fixture is entirely separate from the research output.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
python -m pytest -q tests/tifs
export TIFS_PARITY_REGISTRY="$TIFS_ASSETS/assets.json" TIFS_PARITY_VIDEO="$TIFS_INVENTORY/parity-single-face.mp4"
export TIFS_MFFI_RELEASE="$TIFS_ASSETS/mffi-models"
python -m pytest -q tests/tifs/test_external_release.py tests/tifs/test_learning_release.py
python -m tifs.fixture "$TIFS_ROOT/engineering-fixture"
python -m tifs.campaign all --config "$TIFS_ROOT/engineering-fixture/fixture.yaml"
```

Return to Shell Access with the environment and path exports above. Submit once. Use an appropriate wall time/partition within the site's allocation; the defaults reserve 8 CPUs, 64 GiB, 48 hours per job, and one GPU for a model or crop task. `--parallel 2` caps concurrent array members. Every job uses the same checkout, inventory and environment.

```bash
CFG=configs/tifs/campaign.yaml
python -m tifs.submit --config "$CFG" --parallel 2 --dry-run
python -m tifs.submit --config "$CFG" --parallel 2
squeue -u "$USER"
```

The dependency graph runs `prepare → clean references → remaining model jobs → analyze → crop → report`. Logs are under `$TIFS_OUTPUT/logs/`. If an allocation expires, correct the resource problem, wait for the old workers to end or cancel your superseded jobs, and submit the identical graph again. Preparation reuses cached media and per-view frame stacks. Completed model jobs verify their receipts. Interrupted training resumes from its last complete epoch. External reference scoring checkpoints at most every 25 renditions. A changed protocol, code, environment or input registry needs a new output directory. Never tune on the sealed external results. To inspect a failure in an allocation, the same stage is directly callable, e.g. `python -m tifs.campaign model --config "$CFG" --run dino-paired__s42`. For a single allocated GPU, `python -m tifs.campaign all --config "$CFG"` executes the same workflow sequentially.

## Cost and artifacts

There are **39 trained models** (13 specifications, three seeds) and **four frozen comparators**. Per balanced epoch, count the actual sampled recording draws in `training_history.json`. A run with 1,400 draws, eight epochs, four sampled frames and three training routes uses 134,400 frame views. Clean arms use one route. At that illustrative enrollment, the full training grid processes about 4.7 million frame views. At a measured effective rate of 10–50 training views/s this would cost roughly 26–131 single-GPU hours, plus inference, activation checkpoint overhead if excluded from timing, preflight, and media processing. This is a planning scenario. Measure a matched training batch, full-pipeline epoch, and LipForensics landmark extraction on a development-only pilot before reserving the campaign. Large external cohorts can make repeated inference the dominant cost. Each cached rendition uses roughly 1.15 MiB for eight 224×224 RGB frames before raw media and diagnostic crops. Provision checkpoints, score journals, empirical source/received bitrates, and temporary space separately. Preserve sufficient quota for atomic checkpoint replacement.

| Location under `$TIFS_OUTPUT` | What to use it for |
|---|---|
| `enrollment/`, `protocol.json`, `asset_probe.json` | Independent-unit power, sealed settings/revision/inputs and strict model loading |
| `panel/` | Registered view keys, original/capture lineage, file digests, crop failures, exclusions, duplication, alignment and power audits |
| `runs/<model>__s<seed>/` | Identity, model/score receipt, predictions, checkpoint, training history or frozen detector score journal |
| `analysis/` | Frozen cutoffs, coverage, concordance and failure counts, route/clean/joint metrics, recalibration ceilings, controls, intervals, selection and replication decisions |
| `crop/` | Same-key, same-checkpoint, same-cutoff tracked-face analysis and paired differences |
| `paper/` | Start with `SUMMARY.md`. Full-precision tables, LaTeX fragments, vector PDFs and `ARTIFACTS.json` accompany it |

See [detector training](tifs_models.md) and [inference and uncertainty](tifs_statistics.md) for interpretation. Retain sealed ledgers and artifacts with the submission. Independent curators must verify that the supplied identity keys, generator labels, consent, actual receiver routes, and training-exposure declarations match the field data.

The [FF++ Social archive check](tifs_archive.md) scores linked source dev/cal archive copies using the primary experiment’s frozen checkpoints and thresholds. Its separate descriptive output preserves the distinction from external confirmation.
