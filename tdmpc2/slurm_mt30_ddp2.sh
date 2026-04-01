#!/bin/bash -l
#SBATCH --job-name=mt30_ddp2
#SBATCH --partition=tide
#SBATCH --qos=epic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -e mt30_ddp2.err
#SBATCH -o mt30_ddp2.out

set -euo pipefail

export HOME=/scratch_tide/wy524
export NETRC=$HOME/.netrc
mkdir -p "$HOME"
touch "$NETRC"
chmod 600 "$NETRC"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export WANDB_API_KEY="7feaca49acb80e68486cc6e9c40b2f2c397a0fae"
export WANDB_MODE=online

source /scratch_tide/wy524/miniconda3/etc/profile.d/conda.sh
conda activate tdmpc2

cd /scratch_tide/wy524/tdmpc2/tdmpc2/

# Resolve rank/world env vars from torchrun.
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=2 train.py \
  --config-name DDP_config \
  task=mt30 \
  model_size=19 \
  batch_size=1024 \
  steps=10000000 \
  planner_type=diffusion \
  diffusion_steps=20 \
  diffusion_num_samples=512 \
  diffusion_num_elites=64 \
  diffusion_num_pi_trajs=24 \
  diffusion_clamp_each_step=false \
  eval_episodes=5 \
  eval_freq=0 \
  save_model_every=2500000 \
  ddp_compile_strategy=off \
  compile=true \
  compile_mode='max-autotune-no-cudagraphs' \
  diffusion_eval_compile=false \
  seed=5019
