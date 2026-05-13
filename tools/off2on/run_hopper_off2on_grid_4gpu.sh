#!/bin/bash
set -eo pipefail

# ===== Runtime env =====
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin

# ===== conda =====
set +u
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate tdmpc2-mt80
set -u

# ===== Run config =====
NUM_GPUS=${NUM_GPUS:-4}
CPUS_PER_GPU=${CPUS_PER_GPU:-12}
RESERVED_CPUS=${RESERVED_CPUS:-14}
OMP_THREADS=${OMP_THREADS:-8}

# offline->online common settings
STEPS=${STEPS:-40000}
EVAL_FREQ=${EVAL_FREQ:-1000}
EVAL_EPISODES=${EVAL_EPISODES:-10}
MODEL_SIZE=${MODEL_SIZE:-19}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-true}

# paths
REPO_ROOT=${REPO_ROOT:-/media/damoxing/che-liu-fileset/cxy_worldmodel/tdmpc2/tdmpc2}
CHECKPOINT=${CHECKPOINT:-/media/datasets/cheliu21/cxy_worldmodel/checkpoint/mt80/19/final.pt}
SAVE_PATH=${SAVE_PATH:-/media/datasets/cheliu21/cxy_worldmodel/off2on_outputs}

TASK_NAME=${TASK_NAME:-"Hopper Hop"}
SEEDS=(7 8 9 10 11 12 13 14 15 16 17 18)
CONTRASTIVE_BETAS=(0.1 0.01)
DIFFUSION_STEPS_LIST=(5 20)

cd "${REPO_ROOT}"

RUN_TAG="off2on_hopperhop_grid4_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${SAVE_PATH}/${RUN_TAG}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

TOTAL_BOUND_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
TOTAL_NEEDED_CPUS=$((TOTAL_BOUND_CPUS + RESERVED_CPUS))

echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}, CPUS_PER_GPU=${CPUS_PER_GPU}, RESERVED_CPUS=${RESERVED_CPUS}"
echo "[INFO] OMP_THREADS=${OMP_THREADS}, nproc=$(nproc)"
echo "[INFO] CHECKPOINT=${CHECKPOINT}"
echo "[INFO] SAVE_PATH=${SAVE_PATH}"
echo "[INFO] TASK_NAME=${TASK_NAME}"
echo "[INFO] STEPS=${STEPS}, EVAL_FREQ=${EVAL_FREQ}, EVAL_EPISODES=${EVAL_EPISODES}"
echo "[INFO] CONTRASTIVE_BETAS=${CONTRASTIVE_BETAS[*]}"
echo "[INFO] DIFFUSION_STEPS_LIST=${DIFFUSION_STEPS_LIST[*]}"
echo "[INFO] SEEDS(${#SEEDS[@]}): ${SEEDS[*]}"
echo "[INFO] TOTAL_BOUND_CPUS=${TOTAL_BOUND_CPUS}, TOTAL_NEEDED_CPUS=${TOTAL_NEEDED_CPUS}"

# build 4 parameter groups for 4 GPUs
declare -a JOB_BETAS
declare -a JOB_DIFF_STEPS
for BETA in "${CONTRASTIVE_BETAS[@]}"; do
  for DSTEP in "${DIFFUSION_STEPS_LIST[@]}"; do
    JOB_BETAS+=("${BETA}")
    JOB_DIFF_STEPS+=("${DSTEP}")
  done
done
NUM_JOBS=${#JOB_BETAS[@]}

if [ "${NUM_GPUS}" -ne "${NUM_JOBS}" ]; then
  echo "[ERROR] NUM_GPUS=${NUM_GPUS}, but need exactly ${NUM_JOBS} GPUs for 1 param-group per GPU."
  exit 1
fi

echo "[INFO] Total parameter groups=${NUM_JOBS}"

declare -a pids
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
(
    set -eo pipefail

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export OMP_NUM_THREADS=${OMP_THREADS}
    export MKL_NUM_THREADS=${OMP_THREADS}
    export OPENBLAS_NUM_THREADS=${OMP_THREADS}
    export NUMEXPR_NUM_THREADS=${OMP_THREADS}

    CPU_START=$((GPU_ID * CPUS_PER_GPU))
    CPU_END=$((CPU_START + CPUS_PER_GPU - 1))
    CPU_RANGE="${CPU_START}-${CPU_END}"

    BETA="${JOB_BETAS[$GPU_ID]}"
    DSTEP="${JOB_DIFF_STEPS[$GPU_ID]}"

    declare -a gpu_job_pids
    for SEED in "${SEEDS[@]}"; do
      JOB_DIR="${RUN_ROOT}/gpu${GPU_ID}/beta${BETA}_diff${DSTEP}/seed${SEED}"
      mkdir -p "${JOB_DIR}"

      echo "[START] GPU=${GPU_ID} TASK='${TASK_NAME}' BETA=${BETA} DIFF_STEPS=${DSTEP} SEED=${SEED} CPU_RANGE=${CPU_RANGE}"

      (
        set -eo pipefail
        taskset -c "${CPU_RANGE}" python -u off2on.py \
          task=mt80 \
          obs=state \
          model_size=${MODEL_SIZE} \
          seed=${SEED} \
          checkpoint="${CHECKPOINT}" \
          load_checkpoint=${LOAD_CHECKPOINT} \
          save_path="${SAVE_PATH}" \
          off2on_task="${TASK_NAME}" \
          steps=${STEPS} \
          eval_freq=${EVAL_FREQ} \
          eval_episodes=${EVAL_EPISODES} \
          contrastive_beta=${BETA} \
          diffusion_steps=${DSTEP} \
          planner_type=diffusion \
          enable_wandb=false \
          save_video=false \
          save_agent=false \
          > "${JOB_DIR}/stdout.log" 2>&1
      ) &
      gpu_job_pids+=($!)
      sleep 2
    done

    GPU_FAIL=0
    for job_pid in "${gpu_job_pids[@]}"; do
      if ! wait "${job_pid}"; then
        GPU_FAIL=1
      fi
    done
    if [ "${GPU_FAIL}" -ne 0 ]; then
      echo "[WARN] GPU=${GPU_ID} one or more jobs failed."
      exit 1
    fi
    echo "[DONE]  GPU=${GPU_ID} all assigned jobs completed."
) 2>&1 | sed "s/^/[GPU${GPU_ID}] /" | tee "${LOG_ROOT}/gpu${GPU_ID}.log" &

  pids+=($!)
  sleep 5
done

echo "[INFO] All GPU workers launched. PID list: ${pids[*]}"
echo "[INFO] Reserved CPU range: ${TOTAL_BOUND_CPUS}-$((TOTAL_NEEDED_CPUS-1))"

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

echo "[DONE] all hopper-hop off2on grid jobs finished."
echo "[CHECK] outputs are under: ${SAVE_PATH}/hopper-hop/ (named by off2on.py)"
