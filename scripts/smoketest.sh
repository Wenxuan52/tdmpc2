#!/bin/bash -l
#SBATCH --job-name=tdmpc2_smoke
#SBATCH --partition=tide
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH -e tdmpc2_smoke.err
#SBATCH -o tdmpc2_smoke.out

set -euo pipefail

# writable home + netrc on scratch (avoid /home)
export HOME=/scratch_tide/wy524
export NETRC=$HOME/.netrc
mkdir -p "$HOME"
touch "$NETRC"
chmod 600 "$NETRC"

# mujoco egl
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# wandb auth (don’t rely on /home)
export WANDB_API_KEY="7feaca49acb80e68486cc6e9c40b2f2c397a0fae"
export WANDB_MODE=online

# conda
source /scratch_tide/wy524/miniconda3/etc/profile.d/conda.sh
conda activate tdmpc2

cd /scratch_tide/wy524/tdmpc2
python -u tdmpc2/train.py task=dog-run steps=10000
