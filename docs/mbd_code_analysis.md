# Diffusion Planner Code Analysis (`mbd/planners/mbd_planner.py`)

## Scope inspected
- Primary: `mbd/planners/mbd_planner.py`.
- Direct dependencies used by planner loop:
  - `mbd/utils.py` (`rollout_us`, `render_us`).
  - `mbd/envs/__init__.py` (`get_env` dispatch).
  - Demo-related env hooks: `eval_xref_logpd`, `rew_xref` (e.g., `mbd/envs/car2d.py`, `mbd/envs/humanoidtrack.py`).

---

## A) Planner overview (what it does)

### High-level algorithm (as implemented)
- Initializes an environment and jit-compiles `env.step`, `env.reset`, and rollout utility (`rollout_us`).
- Builds a linear beta schedule (`beta0 -> betaT`) over `Ndiffuse` steps, then derives `alpha=1-beta`, `alpha_bar=cumprod(alpha)`, and `sigma_t = sqrt(1-alpha_bar_t)`.
- Maintains a trajectory variable over reverse steps (`i = Ndiffuse-1 ... 1`) in a deterministic reverse process.
- At each reverse step:
  1. Forms noisy latent `Y_i = sqrt(alpha_bar_i) * Ybar_i`.
  2. Samples `Nsample` candidate clean trajectories `Y0s ~ q(Y0 | Y_i)` via Gaussian perturbation around `Ybar_i`.
  3. Clips each action to `[-1, 1]`.
  4. Rolls out each candidate through true env dynamics (`env.step`) to get per-step rewards (and pipeline states).
  5. Converts returns to logits (`logp0`) by standardization + temperature scaling.
  6. Softmax weights over candidates.
  7. Computes weighted mean trajectory `Ybar = sum_n w_n * Y0s_n`.
  8. Computes a score-like term and updates `Y_{i-1}`.
  9. Converts to next carried `Ybar_{i-1}`.
- Final output trajectory is `Yi[-1]` from the collected reverse sequence, and final reward is measured with one more rollout.

### What is “model-based” here?
- Candidate trajectories are scored by **actual environment rollouts** (`rollout_us`), i.e., repeatedly applying `env.step` over horizon `Hsample`.
- Rewards come from environment dynamics + reward function, not from a trained denoiser network.
- So guidance signal is model-based return estimation over sampled open-loop action sequences.

---

## B) Exact reverse diffusion math used in code

Below uses planner variable names (`MBDPlanner.plan()` equivalent is `run_diffusion()` + inner `reverse_once()`).

### Symbols in code
- `i`: reverse diffusion index (`Ndiffuse-1` down to `1`).
- `Ybar_i`: carried trajectory variable at step `i` (shape `[H, Nu]`).
- `Y_i`: noisy trajectory at step `i`.
- `Y0s`: sampled candidate clean trajectories (`Nsample` of them).
- `beta_i`: from linear schedule between `beta0`, `betaT`.
- `alpha_i = 1 - beta_i`.
- `alpha_bar_i = prod_{k<=i} alpha_k`.
- `sigma_i = sqrt(1 - alpha_bar_i)`.

### Candidate sampling `q(Y0 | Y_tau)` (code form)
- `eps_u ~ N(0, I)` with shape `[Nsample, Hsample, Nu]`.
- `Y0s = Ybar_i + sigma_i * eps_u`.
- `Y0s = clip(Y0s, -1, 1)`.

### Return computation
- `rewss, qs = vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)`.
  - `rollout_us` runs `lax.scan(step_env, state, us)` for one trajectory.
  - `rewss` shape: `[Nsample, Hsample]`.
- `rews = mean(rewss, axis=-1)` -> per-sample scalar return (mean reward over horizon).

### Weight/logit computation
- `rew_std = std(rews)`; if `<1e-4`, replace with `1.0`.
- `rew_mean = mean(rews)`.
- Base logits:
  - `logp0 = (rews - rew_mean) / rew_std / temp_sample`.
- Optional demo guidance (`enable_demo=True`):
  - `xref_logpds = vmap(env.eval_xref_logpd)(qs)`.
  - Shift: `xref_logpds -= max(xref_logpds)`.
  - `logpdemo = (xref_logpds + env.rew_xref - rew_mean) / rew_std / temp_sample`.
  - Replace where better: `logp0 = where(logpdemo > logp0, logpdemo, logp0)`.
  - Re-standardize logits: `logp0 = (logp0 - mean(logp0)) / std(logp0) / temp_sample`.
- Final weights:
  - `w = softmax(logp0)` over sample dimension.

### Weighted trajectory mean
- `Ybar = einsum("n,nij->ij", w, Y0s)`.

### Score computation (exact code)
- First reconstruct noisy state from carried variable:
  - `Y_i = sqrt(alpha_bar_i) * Ybar_i`.
- Score-like term:
  - `score_i = ( -Y_i + sqrt(alpha_bar_i) * Ybar ) / (1 - alpha_bar_i)`.

### Reverse update
- `Y_{i-1} = ( Y_i + (1 - alpha_bar_i) * score_i ) / sqrt(alpha_i)`.
- Carry conversion:
  - `Ybar_{i-1} = Y_{i-1} / sqrt(alpha_bar_{i-1})`.

### Important note on simplification
- Substituting `score_i` into update gives:
  - `Y_{i-1} = Ybar / sqrt(alpha_i)`.
- Then carried value:
  - `Ybar_{i-1} = Ybar / sqrt(alpha_i * alpha_bar_{i-1}) = Ybar / sqrt(alpha_bar_i)`.
- Since `Y_i = sqrt(alpha_bar_i) * Ybar_i`, this effectively makes reverse step deterministic toward weighted candidate mean; no extra reverse noise is injected.
- `Sigmas_cond`/`sigmas_cond` are computed but not used in update.

### Clipping / bounds / normalization
- Planner clips sampled actions `Y0s` to `[-1, 1]`.
- Some envs also clip actions in `env.step` (e.g., `car2d`).
- Return normalization uses per-step batch stats (`mean/std`) and std floor at `1e-4`.

---

## C) Shapes and batching

### Typical shapes
- `Nu`: action dimension (`env.action_size`).
- `Hsample`: planning horizon.
- `Ybar_i`, `Y_i`, `Ybar`, `Y_{i-1}`: `[Hsample, Nu]`.
- `eps_u`, `Y0s`: `[Nsample, Hsample, Nu]`.
- `rewss`: `[Nsample, Hsample]`.
- `rews`, `logp0`, `weights`: `[Nsample]`.
- `qs`: `[Nsample, Hsample, ...state_shape...]` (pipeline state trees; shape depends on env).

### Batch/time conventions
- Trajectory tensors are **time-first within each sample**: `[H, A]`.
- Candidate batch is extra leading dim: `[N, H, A]`.
- `rollout_us` scans over time axis of one `[H, A]` trajectory.
- `vmap(rollout_us)` batches over candidate dimension `N`.

---

## D) Critical hyperparameters & defaults

### Defaults (`Args`)
- `Nsample=2048` (up to 8192 recommended for `humanoidrun`).
- `Hsample=50` (40 for `pushT` recommendation).
- `Ndiffuse=100` (200 for `pushT`, 300 for `humanoidrun`).
- `temp_sample=0.1` (env-dependent recommended override map).
- `beta0=1e-4`, `betaT=1e-2` (linear schedule).
- `enable_demo=False` by default.

### Stability knobs
- Reward std floor: if `std < 1e-4`, use `1.0`.
- Softmax temperature via `temp_sample`.
- Optional extra logit re-standardization under demo mode.
- Action clipping to `[-1,1]` before rollout.

### Computational bottlenecks
- Dominant cost: `Ndiffuse * Nsample * Hsample` environment steps.
- `vmap(rollout_us)` + `lax.scan` is JAX-friendly but still heavy at large `Nsample`.
- Memory pressure from storing per-sample pipeline states `qs` when demo mode is enabled.

---

## E) Integration notes for porting to TD-MPC2

### Planner logic that ports almost as-is
- Beta/alpha schedule creation.
- Candidate generation around current mean trajectory.
- Return-to-logit normalization + temperature softmax weighting.
- Weighted mean action sequence update.
- Deterministic reverse recursion (or simplified direct update).

### Repo-specific pieces to replace
- JAX env API (`env.reset`, `env.step`, Brax `State`/pipeline_state).
- `rollout_us` based on true environment stepping.
- Demo hooks tied to env internals (`eval_xref_logpd`, `rew_xref`, pipeline state layouts).
- Rendering / file output pieces.

### Adapter layer required for TD-MPC2
- Core scorer API:
  - `G(z0, action_seq_batch) -> score_batch`
  - Input: latent start state `z0`, actions `[N, H, A]`.
  - Output: scalar score per sample `[N]`.
- Inside `G` (TD-MPC2 version):
  - Encode observation -> latent `z` (if needed).
  - Roll latent dynamics over `H` with action sequence.
  - Accumulate reward predictions and optionally terminal value/Q bootstrap.
  - Return scalar objective aligned with planner (mean/sum discounted return; choose explicitly).
- Action handling:
  - Keep clip/mask projection (continuous bounds, actuator masks, optional squashing).
- Warm start prior:
  - Provide initial `Ybar_N` from previous MPC solution shift (instead of zeros) for receding-horizon use.

### Suggested port decisions
- Preserve score normalization and softmax weighting first (behavioral fidelity).
- Then test alternatives: elite selection/CEM-style truncation, different temp schedules, additive reverse noise.

---

## F) Gotchas / tricks in current code

- `Sigmas_cond` and `sigmas_cond` are computed but unused (possible leftover from stochastic DDPM-style update).
- Reverse update currently deterministic; VERIFY whether authors intended stochastic sampling at each reverse step.
- Returns use **mean over horizon** (`rews.mean(axis=-1)`), not sum/discounted sum.
- Action clipping happens at planner sample generation and may also happen in env step.
- Demo mode mixes reward logits and reference-likelihood logits; scales by same `rew_std/temp_sample`.
- Demo mode does `max` subtraction on `xref_logpds` and a second standardization pass.
- No elite top-k truncation; all samples contribute via softmax weights.
- Assumes continuous action space compatible with Gaussian perturbation + clipping.
- Planner is open-loop over fixed horizon from reset state; no receding-horizon replan loop in this file.

---

## Quick code map (for porting work)
- `run_diffusion()` in `mbd/planners/mbd_planner.py`:
  - schedule setup block,
  - `reverse_once()` score/weight/update block,
  - `reverse()` loop from `Ndiffuse-1` to `1`.
- `rollout_us()` in `mbd/utils.py`: per-trajectory rollout primitive.
- `get_env()` in `mbd/envs/__init__.py`: environment factory.
- Demo compatibility examples:
  - `mbd/envs/car2d.py` (`eval_xref_logpd`, `rew_xref`, action clipping).
  - `mbd/envs/humanoidtrack.py` (`eval_xref_logpd`, `rew_xref`).