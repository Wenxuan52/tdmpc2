#!/bin/bash -l

#SBATCH --job-name=envtest
#SBATCH --partition=tide
#SBATCH --qos=long
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e envtest.err
#SBATCH -o envtest.out

set -euo pipefail

source /scratch_tide/wy524/miniconda3/etc/profile.d/conda.sh
conda activate tdmpc2

export HF_HOME=/scratch_tide/wy524/hf
export HF_HUB_CACHE=/scratch_tide/wy524/hf/hub
export XDG_CACHE_HOME=/scratch_tide/wy524/.cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME"

cd /scratch_tide/wy524/tdmpc2/tdmpc2_data

python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="nicklashansen/tdmpc2",
    repo_type="dataset",
    allow_patterns=["mt30/*"],
    local_dir=".",
)
print("snapshot_download path:", path)
print("mt30 download finished")
PY

echo "=== Check files ==="
ls -lh mt30

