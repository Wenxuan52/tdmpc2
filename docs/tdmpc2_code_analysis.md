# TD-MPC2 Code Map (Updated for Diffusion Step1 + Step2A + Step2B)

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
- MT30 task list: `tdmpc2/common/__init__.py::TASK_SET['mt30']`.
- `tdmpc2/common/parser.py::parse_cfg` sets:
  - `cfg.multitask = cfg.task in TASK_SET`
  - `cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])`
- `tdmpc2/envs/__init__.py::make_multitask_env` builds one env per task and stores:
  - `cfg.obs_shapes`, `cfg.action_dims`, `cfg.episode_lengths`.
- Offline MT30 eval:
  - `tdmpc2/trainer/offline_trainer.py::eval` loops task index and calls `agent.act(..., task=task_idx)`.

---

## B) Core modules and current planner structure

### Agent and planning dispatch
- **File:** `tdmpc2/tdmpc2.py`
- **Class:** `TDMPC2`
- **Planning selection:**
  - `act(obs, t0, eval_mode, task)` routes to `self.plan(...)` when `cfg.mpc=True`.
  - Planner binding supports MPPI and diffusion (`planner_type`), with diffusion implemented in a dedicated module.
- **Value helper:** `_estimate_value(z, actions, task)` remains the fast no-grad trajectory scorer used by planners.

### Diffusion planner module (implemented)
- **File:** `tdmpc2/planners/diffusion_planner.py`
- **Class:** `DiffusionPlanner`
- **Main method:** `plan(agent, obs, t0=False, eval_mode=False, task=None)`.
- **Core Step1 behavior:**
  - Reverse diffusion over action sequence `x_tau` using schedule (`beta`, `alpha`, `alpha_bar`).
  - Samples candidates `a0_samples [N,H,A]`, scores with `_estimate_value`, builds weighted mean `a_bar`, computes model-based score `score_mb`, then deterministic reverse update.
  - Maintains warm start via `agent._prev_mean` shift and clamps/masks actions.

### World model APIs used by diffusion guidance
- **File:** `tdmpc2/common/world_model.py`
- Used calls:
  - `encode(obs, task) -> z0`
  - `next(z, a, task) -> z_next`
  - `reward(z, a, task) -> logits` (scalarized via `common.math.two_hot_inv`)
  - `Q(z, a, task, return_type={'min','avg','all'}, target=..., detach=...)`
  - `pi(z, task)` for terminal bootstrap action.

---

## C) Diffusion planner details (Step1/Step2A/Step2B)

### Step1: model-based no-grad diffusion planning
At each reverse step `tau`:
1. Build conditional sample distribution around current `x_tau`.
2. Sample `N` trajectories `a0_samples [N,H,A]`, apply multitask mask, clamp `[-1,1]`.
3. Score candidates with `_estimate_value` (no grad path).
4. Softmax-weight candidates to get `a_bar [H,A]`.
5. Compute model-based score:
   - `score_mb = (-x_tau + sqrt(alpha_bar_tau)*a_bar) / (1 - alpha_bar_tau + eps)`.
6. Reverse update:
   - `x_tau <- (x_tau + (1-alpha_bar_tau)*score) / sqrt(alpha[tau])` (with `score=score_mb` when MF guidance off).

### Step2A: `qgrad_first` guidance (first action only)
Enabled only when:
- `diffusion_mf_guidance=true`, `diffusion_mf_beta>0`, `diffusion_mf_mode=qgrad_first`.

Per reverse step:
1. Create leaf action for first timestep from weighted mean:
   - `a0 = a_bar[0].detach().clone().requires_grad_(True)` inside `torch.enable_grad()`.
2. Compute Q and gradient:
   - `q = model.Q(z0.detach(), a0.unsqueeze(0), ...)`
   - `grad_a0 = autograd.grad(q.sum(), a0)`.
3. Stabilize (`scale`, optional norm clip, finite guard).
4. Construct `score_mf` with only timestep-0 nonzero, apply action mask.
5. Mix with model-based score:
   - `score = (1-beta)*score_mb + beta*score_mf`.

### Step2B: `gradG_elite` guidance (full sequence gradient)
Enabled only when:
- `diffusion_mf_guidance=true`, `diffusion_mf_beta>0`, `diffusion_mf_mode=gradG_elite`.

Per reverse step:
1. Use no-grad candidate values (`values`) to select topK elite indices.
2. Build `a0_elite [K,H,A]`, apply mask+clamp, then **recreate leaf inside `torch.enable_grad()`**:
   - `a0_elite = a0_elite.detach().clone().requires_grad_(True)`.
3. Compute differentiable elite returns via local helper:
   - `DiffusionPlanner._estimate_value_grad_elite(...)`
   - rollout: `reward + dynamics` for horizon, terminal bootstrap with `pi(z_H)` and `Q`.
4. Build elite weights from standardized `G_elite`; detach weights (Step2B requirement).
5. Objective:
   - `obj = sum_i w_i * (eta * G_i)`.
6. Gradient:
   - `grad_elite = autograd.grad(obj, a0_elite)`
   - `grad_bar = grad_elite.sum(dim=0)`
   - `score_mf = grad_bar / sqrt(alpha_bar_tau)`.
7. Apply finite guard, optional grad norm clip, mask, and mix with `score_mb`.

### Gradient-safety and invariants
- Planner entry remains `@torch.no_grad()` for baseline speed.
- MF-guided gradients are computed in local `torch.enable_grad()` scopes.
- Uses `torch.autograd.grad(...)` only; does **not** call `.backward()`; no optimizer interaction.
- Final returned env action remains clamped to `[-1,1]` and masked in multitask.

---

## D) Contracts to preserve

### Inputs
- Callable from `TDMPC2.act(obs, t0, eval_mode, task)`.
- Supports single-task and multitask (`task` index + action mask).
- Uses receding-horizon warm start (`agent._prev_mean`).

### Outputs
- Returns one action tensor `[action_dim]` on device; caller moves to CPU.
- Action is valid for env bounds (clamped) and masked in multitask.

### No behavior change gates
- `planner_type=mppi`: diffusion code path not used.
- `planner_type=diffusion` + `diffusion_mf_guidance=false`: pure Step1 behavior.
- MF guidance activates only when corresponding mode is selected and `diffusion_mf_beta>0`.

---

## E) Config inventory (current)

### Existing diffusion planning keys
- `diffusion_steps`
- `diffusion_beta0`, `diffusion_betaT`
- `diffusion_num_samples`
- `diffusion_temperature`
- `diffusion_action_noise`
- `diffusion_log_stats`

### Step2A / Step2B MF-guidance keys
- `diffusion_mf_guidance`
- `diffusion_mf_mode` (`qgrad_first` | `gradG_elite`)
- `diffusion_mf_beta`
- `diffusion_mf_q_type` (`min` | `avg`)
- `diffusion_mf_use_target_q`
- `diffusion_mf_scale` (Step2A)
- `diffusion_mf_norm_clip` (Step2A)
- `diffusion_mf_eta` (Step2B)
- `diffusion_mf_topk` (Step2B)
- `diffusion_mf_temp` (Step2B)
- `diffusion_mf_std_floor` (Step2B)
- `diffusion_mf_grad_norm_clip` (Step2B)
- `diffusion_mf_detach_weights` (Step2B; expected true)

### Recommended initial ranges for Step2B sweeps
- `diffusion_mf_beta`: `0.05 ~ 0.2`
- `diffusion_mf_topk`: `16 ~ 32`
- `diffusion_mf_temp`: `0.1`
- `diffusion_steps`: `10`
- `diffusion_num_samples`: `32 ~ 64`

---

## F) Quick run-mode examples

- **Step1 baseline diffusion**:
  - `planner_type=diffusion diffusion_mf_guidance=false`
- **Step2A**:
  - `planner_type=diffusion diffusion_mf_guidance=true diffusion_mf_mode=qgrad_first diffusion_mf_beta=0.1`
- **Step2B**:
  - `planner_type=diffusion diffusion_mf_guidance=true diffusion_mf_mode=gradG_elite diffusion_mf_beta=0.1 diffusion_mf_topk=16 diffusion_mf_temp=0.1 diffusion_mf_eta=1.0`
