#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------
# Packed multi-size MT80 launcher for low-utilization small models.
#
# Strategy:
# - Each GPU runs multiple independent jobs concurrently.
# - Small models (1M/5M) get more replicas per GPU than large models.
# - Optional CUDA MPS is enabled by default to improve concurrent kernel use.
#
# Default per-GPU packing (total 7 jobs/GPU):
#   model_size=1  -> 3 replicas
#   model_size=5  -> 2 replicas
#   model_size=19 -> 1 replica
#   model_size=48 -> 1 replica
# ------------------------------------------------------------

# ==== environment bootstrap (customize if needed) ====
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin"

set +u
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate tdmpc2-mt80
set -u

# ==== run config ====
TASK=${TASK:-mt80}
BASE_SEED=${BASE_SEED:-31700}
STEPS=${STEPS:-10000000}
SAVE_EVERY=${SAVE_EVERY:-2500000}
NUM_GPUS=${NUM_GPUS:-8}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}

# NOTE: Keep this as your local repo path on cluster.
REPO_ROOT=${REPO_ROOT:-/media/damoxing/che-liu-fileset/cxy_worldmodel/tdmpc2/tdmpc2}
DEST_ROOT=${DEST_ROOT:-/media/datasets/cheliu21/cxy_worldmodel/checkpoint}
DATA_DIR=${DATA_DIR:-/media/datasets/cheliu21/cxy_worldmodel/tdmpc2/mt80}

# CPU allocation
CPUS_PER_GPU=${CPUS_PER_GPU:-19}
RESERVED_CPUS=${RESERVED_CPUS:-23}

# Packed process profile per GPU
REPLICA_1M=${REPLICA_1M:-3}
REPLICA_5M=${REPLICA_5M:-2}
REPLICA_19M=${REPLICA_19M:-1}
REPLICA_48M=${REPLICA_48M:-1}

# Thread controls per process (smaller is usually better when oversubscribing GPU)
OMP_THREADS=${OMP_THREADS:-2}

# Diffusion planner settings
DIFF_STEPS=${DIFF_STEPS:-20}
DIFF_SAMPLES=${DIFF_SAMPLES:-512}
DIFF_ELITES=${DIFF_ELITES:-64}
DIFF_PI_TRAJS=${DIFF_PI_TRAJS:-24}
OFFLINE_DATA_MODE=${OFFLINE_DATA_MODE:-mmap}

# Startup staggering to reduce compile/IO storms
STAGGER_SECONDS=${STAGGER_SECONDS:-8}

# Optional CUDA MPS for better GPU concurrency
ENABLE_MPS=${ENABLE_MPS:-1}
MPS_PIPE_DIR=${MPS_PIPE_DIR:-/tmp/nvidia-mps}
MPS_LOG_DIR=${MPS_LOG_DIR:-/tmp/nvidia-log}

# ------------------------------------------------------------

declare -a MODEL_SIZES=(1 5 19 48)

declare -A REPLICAS=(
  [1]="${REPLICA_1M}"
  [5]="${REPLICA_5M}"
  [19]="${REPLICA_19M}"
  [48]="${REPLICA_48M}"
)

declare -A BATCH_SIZES=(
  [1]="4096"
  [5]="2048"
  [19]="1024"
  [48]="1024"
)

TOTAL_PROCS_PER_GPU=$((REPLICA_1M + REPLICA_5M + REPLICA_19M + REPLICA_48M))
if [ "${TOTAL_PROCS_PER_GPU}" -le 0 ]; then
  echo "[ERROR] TOTAL_PROCS_PER_GPU must be > 0"
  exit 1
fi

THREADS_PER_PROC=$((CPUS_PER_GPU / TOTAL_PROCS_PER_GPU))
if [ "${THREADS_PER_PROC}" -lt 1 ]; then
  THREADS_PER_PROC=1
fi

TOTAL_BOUND_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
TOTAL_NEEDED_CPUS=$((TOTAL_BOUND_CPUS + RESERVED_CPUS))

RUN_TAG="mt80_multisize_pack_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${DEST_ROOT}/${RUN_TAG}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

cleanup_mps() {
  if [ "${ENABLE_MPS}" = "1" ]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}

start_mps() {
  if [ "${ENABLE_MPS}" != "1" ]; then
    return
  fi
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  nvidia-cuda-mps-control -d
  trap cleanup_mps EXIT
}

launch_one_job() {
  local gpu_id="$1"
  local model_size="$2"
  local replica_idx="$3"
  local global_idx="$4"

  local seed=$((BASE_SEED + gpu_id * 1000 + model_size * 100 + replica_idx))
  local batch_size="${BATCH_SIZES[${model_size}]}"

  local job_root="${RUN_ROOT}/gpu${gpu_id}/m${model_size}/rep${replica_idx}"
  local job_log="${LOG_ROOT}/gpu${gpu_id}_m${model_size}_r${replica_idx}.log"
  mkdir -p "${job_root}"

  local cpu_start=$((gpu_id * CPUS_PER_GPU + global_idx * THREADS_PER_PROC))
  local cpu_end=$((cpu_start + THREADS_PER_PROC - 1))
  local hard_end=$((gpu_id * CPUS_PER_GPU + CPUS_PER_GPU - 1))
  if [ "${cpu_end}" -gt "${hard_end}" ]; then
    cpu_end="${hard_end}"
  fi
  local cpu_range="${cpu_start}-${cpu_end}"

  (
    set -euo pipefail
    cd "${REPO_ROOT}"

    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export OMP_NUM_THREADS="${OMP_THREADS}"
    export MKL_NUM_THREADS="${OMP_THREADS}"
    export OPENBLAS_NUM_THREADS="${OMP_THREADS}"
    export NUMEXPR_NUM_THREADS="${OMP_THREADS}"

    if [ "${ENABLE_MPS}" = "1" ]; then
      export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
      export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
    fi

    echo "[START] gpu=${gpu_id} model_size=${model_size} replica=${replica_idx} seed=${seed} cpu=${cpu_range}"

    taskset -c "${cpu_range}" python -u train.py \
      task="${TASK}" \
      model_size="${model_size}" \
      batch_size="${batch_size}" \
      steps="${STEPS}" \
      planner_type=diffusion \
      diffusion_steps="${DIFF_STEPS}" \
      diffusion_num_samples="${DIFF_SAMPLES}" \
      diffusion_num_elites="${DIFF_ELITES}" \
      diffusion_num_pi_trajs="${DIFF_PI_TRAJS}" \
      diffusion_clamp_each_step=false \
      eval_episodes=5 \
      eval_freq=0 \
      save_model_every="${SAVE_EVERY}" \
      compile=true \
      compile_mode=reduce-overhead \
      data_dir="${DATA_DIR}" \
      offline_data_mode="${OFFLINE_DATA_MODE}" \
      seed="${seed}" \
      offline_checkpoint_root="${job_root}"

    echo "[DONE] gpu=${gpu_id} model_size=${model_size} replica=${replica_idx} seed=${seed}"
  ) 2>&1 | sed "s/^/[GPU${gpu_id}|M${model_size}|R${replica_idx}] /" | tee "${job_log}" &

  pids+=("$!")
}

# ==== main ====

echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] TASK=${TASK}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}"
echo "[INFO] GPU_IDS=${GPU_IDS}"
echo "[INFO] CPUS_PER_GPU=${CPUS_PER_GPU}, RESERVED_CPUS=${RESERVED_CPUS}"
echo "[INFO] TOTAL_BOUND_CPUS=${TOTAL_BOUND_CPUS}, TOTAL_NEEDED_CPUS=${TOTAL_NEEDED_CPUS}"
echo "[INFO] Replica profile per GPU: 1M=${REPLICA_1M}, 5M=${REPLICA_5M}, 19M=${REPLICA_19M}, 48M=${REPLICA_48M}"
echo "[INFO] Total processes per GPU=${TOTAL_PROCS_PER_GPU}, THREADS_PER_PROC=${THREADS_PER_PROC}, OMP_THREADS=${OMP_THREADS}"
echo "[INFO] offline_data_mode=${OFFLINE_DATA_MODE}"
echo "[INFO] visible cpu count: $(nproc)"

start_mps

declare -a pids=()

for gpu_id in ${GPU_IDS}; do
  global_proc_idx=0
  for model_size in "${MODEL_SIZES[@]}"; do
    replicas="${REPLICAS[${model_size}]}"
    if [ "${replicas}" -le 0 ]; then
      continue
    fi
    for ((r=0; r<replicas; r++)); do
      launch_one_job "${gpu_id}" "${model_size}" "${r}" "${global_proc_idx}"
      global_proc_idx=$((global_proc_idx + 1))
      sleep "${STAGGER_SECONDS}"
    done
  done
done

echo "[INFO] All packed jobs launched. count=${#pids[@]}"

echo "[INFO] Reserved CPU range (not bound): ${TOTAL_BOUND_CPUS}-$((TOTAL_NEEDED_CPUS - 1))"

echo "[TIP] Monitor GPU util: watch -n 2 nvidia-smi"

FAIL=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

echo
if [ "${FAIL}" -ne 0 ]; then
  echo "[WARN] Some jobs failed. Check logs under: ${LOG_ROOT}"
  exit 1
fi

echo "[DONE] all packed multi-size jobs finished."

echo
echo "[CHECK] listing results under ${RUN_ROOT}:"
find "${RUN_ROOT}" -maxdepth 4 \( -type f -o -type d \) | sort
