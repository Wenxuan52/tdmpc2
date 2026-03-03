# TD-MPC2 Code Map for Future Diffusion Planner Integration

## A) Repository entry points & runtime flow

### Launch points
- Training entry: `tdmpc2/train.py::train(cfg)` (Hydra `@hydra.main(config_name='config', config_path='.')`).
  - Calls `parse_cfg` to expand model/task defaults and multitask flags.
  - Builds `{env, agent, buffer, logger}` and selects trainer:
    - `OnlineTrainer` for single-task.
    - `OfflineTrainer` for `mt30` / `mt80` multitask.
- Evaluation entry: `tdmpc2/evaluate.py::evaluate(cfg)`.
  - Loads checkpoint into `TDMPC2` and loops tasks/episodes.

### MT30 task set + evaluation wiring
- MT30 task list is in `tdmpc2/common/__init__.py::TASK_SET['mt30']`.
- `tdmpc2/common/parser.py::parse_cfg` sets:
  - `cfg.multitask = cfg.task in TASK_SET`
  - `cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])`
- `tdmpc2/envs/__init__.py::make_multitask_env` builds one env per task and stores:
  - `cfg.obs_shapes`, `cfg.action_dims`, `cfg.episode_lengths`.
- Offline MT30 evaluation path:
  - `tdmpc2/trainer/offline_trainer.py::eval` loops `task_idx in range(len(cfg.tasks))`, then calls `agent.act(..., task=task_idx)`.
- Standalone MT30 eval path:
  - `tdmpc2/evaluate.py::evaluate` loops `task in cfg.tasks` with `task_idx` and reports normalized score.

### End-to-end flow (single loop)
- Env step:
  - `trainer/*.py` gets `obs`, picks `action = agent.act(obs, t0=..., task=...)`, executes `env.step(action)`.
- Buffer write:
  - Online: append transitions into episode TensorDict, commit via `Buffer.add` at episode end.
  - Offline: pre-load dataset with `OfflineTrainer._load_dataset -> Buffer.load`.
- Update:
  - `agent.update(buffer)` -> `Buffer.sample()` -> `TDMPC2._update(...)`:
    - latent rollout consistency loss
    - reward/value/termination losses
    - policy update (`update_pi`)
    - target Q soft update.
- Evaluation:
  - Periodic in trainer (`eval_freq`) + optional standalone `evaluate.py`.

---

## B) Core classes/modules and where things happen

### 1) Main agent (planning + update)
- **File:** `tdmpc2/tdmpc2.py`
- **Class:** `TDMPC2`
- **Purpose:** owns world model, MPC planner, policy/value training.
- **Key methods:**
  - `act(obs, t0=False, eval_mode=False, task=None) -> action[cpu]`
    - If `cfg.mpc=True`: calls `plan` property (`_plan` or compiled version).
    - Else: direct actor action `model.pi(z, task)`.
  - `_plan(obs, t0, eval_mode, task) -> action`
    - MPPI/CEM-like latent planning loop.
  - `_estimate_value(z, actions, task) -> G`
    - rollout return + terminal Q bootstrap (future diffusion planner should reuse this contract).
  - `_update(obs, action, reward, terminated, task=None) -> TensorDict metrics`
    - world-model + critic + policy updates.
- **I/O shapes (as used):**
  - `obs` from buffer sample: `[T+1, B, obs_dim]` (or dict/rgb variant).
  - `action`: `[T, B, action_dim]`, `reward/terminated`: `[T, B, 1]`, `task`: `[B]` (multitask).

### 2) World model components
- **File:** `tdmpc2/common/world_model.py`
- **Class:** `WorldModel`
- **Purpose:** latent dynamics + reward + policy prior + Q ensemble.
- **Submodules:**
  - `_encoder = layers.enc(cfg)`
  - `_dynamics(latent+action+task_emb -> next_latent)`
  - `_reward(latent+action+task_emb -> num_bins)`
  - `_termination(latent+task_emb -> 1)` (if `cfg.episodic`)
  - `_pi(latent+task_emb -> 2*action_dim)` Gaussian head (mean/log_std)
  - `_Qs`: ensemble of Q heads outputting `num_bins` each.
- **Important APIs:**
  - `encode(obs, task) -> z`
  - `next(z, a, task) -> z_next`
  - `reward(z, a, task) -> reward_logits`
  - `Q(z, a, task, return_type={'min','avg','all'}, target=False, detach=False)`
  - `pi(z, task) -> (action, info)`

### 3) Replay / dataset sampling
- **File:** `tdmpc2/common/buffer.py`
- **Class:** `Buffer`
- **Purpose:** episode storage + subsequence sampling via `SliceSampler`.
- **Key methods:**
  - `add(td_episode)` online write.
  - `load(td_batch_episodes)` offline bulk load.
  - `sample() -> (obs, action, reward, terminated, task)` where sequence length is `horizon+1`.
- **Sampling contract:**
  - sampled TensorDict is reshaped/permuted to `[T+1, B, ...]`, then post-processed:
    - `action/reward/terminated` shifted to align as transitions from `obs[t]`.

### 4) Planning internals (MPPI/CEM loop location)
- **File:** `tdmpc2/tdmpc2.py`
- **Function:** `TDMPC2._plan`
- **Purpose:** sample candidate action trajectories, score with `_estimate_value`, refit Gaussian.

### 5) Value estimation helper (future `G`)
- **File:** `tdmpc2/tdmpc2.py`
- **Function:** `TDMPC2._estimate_value`
- **Purpose:** `G = sum_t gamma^t r_t + gamma^H * Q(z_H, pi(z_H))`, with optional termination masking.

### 6) Multi-task action masks + task embeddings
- **Task embedding:** `WorldModel.task_emb` concatenates embedding to obs/latent.
- **Action mask source:** `WorldModel.__init__` creates `_action_masks[task, action_dim]` from `cfg.action_dims`.
- **Mask usage:**
  - In planner `_plan`: masks sampled actions, updated mean/std.
  - In policy `WorldModel.pi`: masks mean/log_std/eps and scales entropy by active action dims.

---

## C) Planning internals (critical)

### Exact action-selection function
- Main call chain: `TDMPC2.act -> TDMPC2.plan(property) -> TDMPC2._plan`.
- `_plan` returns a single env action tensor (clipped to `[-1, 1]`).

### MPPI/CEM-like loop details in `_plan`
- Inputs:
  - `obs[1, obs_dim]` from env step, optional `task` index tensor.
- Pre-seeding with policy trajectories:
  - If `num_pi_trajs>0`, rollout policy prior in latent space to fill first `num_pi_trajs` candidates.
- Distribution state:
  - `mean[horizon, action_dim]` (warm-started by shifted `self._prev_mean` when `t0=False`).
  - `std[horizon, action_dim]` initialized to `max_std`.
- Iteration (`for _ in range(cfg.iterations)`):
  - Sample Gaussian noise `r ~ N(0, I)` for `num_samples - num_pi_trajs` trajectories.
  - Candidate actions: `actions_sample = mean + std * r`, then clip `[-1,1]`.
  - Optional multitask mask multiply.
  - Score each trajectory via `_estimate_value`.
  - Elite selection: `topk(num_elites)`.
  - Weighted update (MPPI-style):
    - `score = exp(temperature * (elite_value - max_elite_value))`, normalized.
    - `mean = weighted_avg(elite_actions)`.
    - `std = sqrt(weighted_var(elite_actions))`, then clamp `[min_std, max_std]`.
- Final action select:
  - sample elite index via `math.gumbel_softmax_sample(score)`.
  - choose first action from selected elite sequence.
  - add exploration noise `+ std[0] * N(0,I)` unless `eval_mode=True`.
  - update warm-start cache `self._prev_mean <- mean`.

### Places tied to Gaussian/MPPI assumptions
- `TDMPC2._prev_mean` buffer shape `[horizon, action_dim]` (receding-horizon shift trick).
- `_plan` assumes Gaussian `mean/std` parameters and per-step std clamp.
- Config knobs tied to MPPI behavior: `iterations`, `num_samples`, `num_elites`, `temperature`, `min_std`, `max_std`, `num_pi_trajs`, `horizon`.
- `_estimate_value` uses `cfg.num_samples` for termination tensor allocation (couples scorer to planner sample count).

---

## D) Interfaces & invariants to preserve when swapping planner

### Planner input contract
- Must be callable from `TDMPC2.act(obs, t0, eval_mode, task)` under `torch.no_grad()`.
- Must accept:
  - encoded obs path (`obs` raw input currently, planner itself calls `model.encode`).
  - optional multitask `task` index tensor.
  - `t0` for episode boundary / warm-start reset behavior.
- Must respect action masks in multitask (`model._action_masks[task]`).

### Planner output contract
- Return one action tensor shape `[action_dim]` on agent device (then `.cpu()` in `act`).
- Action must be clipped to env range `[-1,1]` before env step.
- Preserve/update planner cache for receding-horizon warm start (currently `_prev_mean`).

### Runtime constraints
- Device: all planning tensors on `cuda:0` (`self.device`).
- Batch semantics:
  - candidate rollouts are vectorized across planner sample dimension.
- No-grad inference: planner called under `@torch.no_grad()` from `act`.
- Keep compatibility with optional `torch.compile` through `plan` property.

### Latent rollout API invariants (must reuse)
- Encode: `z0 = model.encode(obs, task)`
- Transition: `z_{t+1} = model.next(z_t, a_t, task)`
- Reward logits -> scalar: `two_hot_inv(model.reward(...), cfg)`
- Terminal bootstrap Q: `model.Q(z_H, a_H, task, return_type='avg')` with `a_H ~ model.pi(z_H, task)`.

---

## E) Implementation hooks for Step1–Step4 (TODO anchors)

### TODO-Step1: Isolate planner interface
- **Modify:** `tdmpc2/tdmpc2.py`
  - Add planner abstraction (e.g., `self.planner`) used by `act`/`plan`.
  - Keep `_estimate_value` reusable as scorer callback.

### TODO-Step2: Add diffusion planner module
- **Likely new file(s):**
  - `tdmpc2/planners/diffusion_planner.py` (new class, reverse denoise loop)
  - `tdmpc2/planners/schedules.py` (beta/sigma/time embeddings/utils)
  - `tdmpc2/planners/__init__.py`
- **Note:** preserve warm-start by caching previous planned sequence (analogue of `_prev_mean`).

### TODO-Step3: Plug diffusion loop in place of MPPI sampling
- **Modify:** `TDMPC2._plan` (or replace by planner class method)
  - Replace Gaussian sampling/refit with reverse diffusion denoising over action trajectories.
  - Keep:
    - horizon semantics
    - receding-horizon shift at `t0=False`
    - action masking for multitask
    - final action extraction from first step.

### TODO-Step4: Future diffusion policy update path
- Current actor update is Gaussian-policy-specific in:
  - `tdmpc2/common/world_model.py::pi`
  - `tdmpc2/tdmpc2.py::update_pi`
- To support diffusion policy training later:
  - add parallel policy head/loss branch while keeping Q/WM losses intact.
  - **VERIFY:** whether to retain current `_pi` as proposal prior for planner hybrid mode.

---

## F) Config parameters inventory

### Planner-related keys (`tdmpc2/config.yaml`)
- `mpc` (toggle planning vs direct policy action)
- `horizon`
- `iterations`
- `num_samples`
- `num_elites`
- `num_pi_trajs`
- `temperature`
- `min_std`, `max_std`

### Actor/value/world-model keys affecting planner integration
- Actor dist: `log_std_min`, `log_std_max`, `entropy_coef`
- Value discretization: `num_bins`, `vmin`, `vmax`, derived `bin_size`
- Discount shaping: `discount_denom`, `discount_min`, `discount_max`
- Loss weights: `reward_coef`, `value_coef`, `consistency_coef`, `termination_coef`, `rho`
- Model scale: `model_size`, `enc_dim`, `mlp_dim`, `latent_dim`, `num_q`, `dropout`, `num_enc_layers`, `simnorm_dim`
- Multitask/task wiring: `task`, `task_dim`, derived `tasks`, `action_dims`, `episode_lengths`, `multitask`

### MT30/MT80 scaling and shape notes
- `parse_cfg` injects model-size presets from `common.MODEL_SIZE`.
- `task_dim` is auto-adjusted for MT30/MT80 edge cases.
- `make_multitask_env` sets max observation/action sizes and per-task true dims; action masks enforce valid per-task action subspace.
- **VERIFY:** episodic termination in multitask paths (`WorldModel.termination` has `assert task is None`), likely safe because MT30/MT80 training here is offline/non-episodic by default.
