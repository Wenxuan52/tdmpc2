# Online Algorithm (Interaction / Data Collection)

## Scope
This document describes only the **online environment interaction loop**: task/episode reset, action generation, environment stepping, and replay write path. It does not describe parameter updates.

## Notation
- \(s_t\): observation at time step \(t\)
- \(z_t = E(s_t)\): latent state from encoder \(E\)
- \(a_{t:t+H-1}\): action sequence over horizon \(H\)
- \(\tau\): diffusion denoising/noise index
- \(F\): latent dynamics, \(R\): reward predictor, \(Q\): critic
- \(\gamma\): discount used inside latent rollout value estimates

## Pseudocode
```text
Algorithm 1: Online interaction loop (single-task or configured multi-task)
Inputs:
  env, replay buffer D, agent (E, F, R, Q, action generator), config

Initialize:
  global_step <- 0
  episode_idx <- 0
  done <- True

while global_step <= max_steps:

  1) If episode ended:
     a) Optionally run periodic evaluation episodes.
     b) If not first episode, push the just-finished trajectory into D.
        Stored fields per step: obs, action, reward, terminated, (task if present).
     c) Reset environment:
        - Single-task: s_0 <- env.reset()
        - Multi-task mode: reset with configured task index if provided by trainer.
     d) Start new local trajectory list with initial item containing s_0.

  2) Choose action source:
     if global_step <= seed_steps:
        a_t <- random action from env
     else:
        call agent.act(s_t, t0 = [first step of episode], eval_mode = False)

  3) Inside agent.act (MPC-enabled path):
     a) Encode current observation: z_t = E(s_t).
     b) Select planning branch by config:
        - MPPI latent planning branch:
            i. Build candidate sequences a_{t:t+H-1}.
           ii. Evaluate candidates via latent rollout using F, R, Q.
          iii. Keep receding-horizon mean/std statistics.
           iv. Return first action a_t.
        - Diffusion-planning branch:
            i. Initialize noisy sequence x^tau (warm-start from previous plan if t0=False).
           ii. For tau = T-1 ... 1:
                 * sample action-sequence candidates conditioned on x^tau
                 * compute model-based correction score_mb from weighted candidate average
                 * optionally add Q-induced score_mf guidance branch
                 * update x^tau -> x^{tau-1}
          iii. Return first action a_t from final denoised sequence.
        - Diffusion-policy branch (direct policy sampling):
            i. sample denoised sequence from epsilon/score network for configured steps
           ii. return first action a_t.
     c) Optional mixed-action mode:
        - choose between planner branch and diffusion-policy branch by scheduled probability.

  4) Environment step:
     (s_{t+1}, r_t, done, info) <- env.step(a_t)

  5) Append one transition record to current trajectory:
     record contains current observation slot + executed action/reward/terminated metadata.

  6) If global_step >= seed_steps:
     run one training update iteration (or a pretrain burst at seed boundary).

  7) global_step <- global_step + 1

end while
```

## Warm-start / receding-horizon state
- MPPI branch shifts the previous mean plan by one step (`prev_mean[1:]`) when `t0=False`.
- Diffusion-planning branch uses the previous denoised sequence as warm-start prior mean when `t0=False`.
- Both branches return only the first action and keep the remaining sequence as an internal prior for the next environment step.

## Config branches (online action selection)
- `mpc`: switch between MPC-style action generation and direct Gaussian actor action.
- `planner_type`: `mppi` vs `diffusion` vs `diffusion_policy` (when policy acting is enabled).
- `diffusion_policy_act`: enables direct sequence sampling from diffusion policy for action execution.
- `student_act_mixing_enabled` + probability schedule flags:
  - `student_act_prob`, `student_act_prob_schedule`, `student_act_prob_warmup_steps`, `student_act_prob_ramp_steps`, `student_act_prob_final`.
- `seed_steps`: before this threshold, actions come from random exploration.
- Diffusion planning controls affecting online sequence generation:
  - `diffusion_steps`, `diffusion_beta0`, `diffusion_betaT`, `diffusion_num_samples`,
    `diffusion_temperature`, `diffusion_action_noise`,
    `diffusion_mf_guidance`, `diffusion_mf_mode`, `diffusion_mf_beta`,
    `diffusion_mf_topk`, `diffusion_mf_eta`, `diffusion_mf_temp`.

## Code pointers
- Entrypoint and trainer dispatch: `tdmpc2/train.py`
- Online loop and replay insertion: `tdmpc2/trainer/online_trainer.py`
- Action-selection call path: `tdmpc2/tdmpc2.py`
- Latent MPPI planning and value rollout: `tdmpc2/tdmpc2.py`
- Diffusion planning and score construction branches: `tdmpc2/planners/diffusion_planner.py`
- Replay storage layout and sampled tensors: `tdmpc2/common/buffer.py`

## VERIFY
- Multi-task online reset policy (which task index is chosen at each episode) should be verified in the trainer+env wrapper integration if you plan to run online multi-task collection.
