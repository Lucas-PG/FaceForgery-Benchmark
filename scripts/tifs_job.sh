#!/bin/bash -l
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch from the checkout}"
test -f tifs/campaign.py || { echo "Submit with --chdir pointing to the checkout" >&2; exit 2; }
source /opt/conda/etc/profile.d/conda.sh
conda activate "${TIFS_ENV:-tifs}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
jobtmp="/scratch/$USER/tifs-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$jobtmp"
export TMPDIR="$jobtmp"
trap 'echo "job=$SLURM_JOB_ID exit=$? elapsed_seconds=$SECONDS at $(date -Is)"' EXIT
echo "job=$SLURM_JOB_ID revision=$(git rev-parse HEAD) node=$(hostname) stage=$2 at $(date -Is)"
cmd=(python -u -m tifs.campaign "$2" --config "$1")
if [[ "$2" == model ]]; then cmd+=(--index "${SLURM_ARRAY_TASK_ID:?Set array index for model job}"); fi
"${cmd[@]}"
