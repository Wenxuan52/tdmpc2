# Offline Algorithm (Learning / Update)

## Scope
This document describes only the **offline update loop** (one optimization iteration): replay sampling, world-model/critic updates, diffusion-policy score-matching update, and logging.

## Notation
- \(s_t\): observation, \(z_t = E(s_t)\): latent state
- \(a^0_{t:t+H-1}\): clean action sequence over horizon \(H\)
- \(a^\tau\): noisy sequence at diffusion step \(\tau\)
- \(F\): latent dynamics, \(R\): reward predictor, \(Q\): critic, \(\gamma\): discount
- \(\bar\alpha_\tau\): cumulative diffusion alpha schedule
- Unified target: \(\text{score}_{\text{target}} = \beta\,\text{score}_{\text{mf}} + (1-\beta)\,\text{score}_{\text{mb}}\)

## Pseudocode (one training iteration)
```text
Algorithm 2: Offline update iteration
Inputs:
  replay buffer D, model params θ = {E, F, R, Q, policy}, target-Q params θ̄, config

A) Sample batch
  1) Sample subsequences of length H+1 from D:
       {s_t, a_t, r_t, done_t}_{t=0..H}
  2) Convert to tensors:
       obs[0:H], action[0:H-1], reward[0:H-1], terminated[0:H-1], optional task id

B) Update world model + critic backbone
  3) Encode next observations:
       z_{t+1}^{enc} = E(obs[t+1])
  4) Build TD targets using bootstrap from target-Q:
       y_t = r_t + γ(1-done_t) * min/avg_Q_target(z_{t+1}^{enc}, π(z_{t+1}^{enc}))
  5) Latent rollout consistency:
       z_0 = E(obs[0])
       z_{t+1}^{pred} = F(z_t^{pred}, a_t)
       L_cons = Σ_t ρ^t ||z_{t+1}^{pred} - z_{t+1}^{enc}||^2
  6) Reward prediction loss:
       L_rew = Σ_t ρ^t * CE_soft(R(z_t^{pred}, a_t), r_t)
  7) Critic loss (ensemble heads):
       L_q = Σ_t ρ^t * CE_soft(Q_i(z_t^{pred}, a_t), y_t)
  8) (If episodic) termination loss:
       L_term = BCE(Termination(z_{t+1}^{pred}), done_t)
  9) Total model loss:
       L_model = c1*L_cons + c2*L_rew + c3*L_term + c4*L_q
 10) Backprop L_model; optimizer step on E/F/R/Q (+ optional task embeddings)

C) Optional Gaussian actor branch
 11) If policy_update_type == gaussian:
       maximize entropy-regularized Q along latent rollout states z_0..z_H

D) Diffusion policy score-matching branch
 12) If policy_update_type == diffusion (or diffusion policy update is enabled):
       a) Draw latent-sequence/action-sequence batch (z_0, a^0) from policy-data buffer.
       b) Sample diffusion index τ ~ Uniform({1,...,T-1}) and noise ε ~ N(0,I).
       c) Forward noise:
            a^τ = sqrt(ᾱ_τ) * a^0 + sqrt(1-ᾱ_τ) * ε
       d) Predict ε_hat = epsilon_net(z_0, a^τ, τ)

 13) Build unified score target for a selected subset (expensive branch):
       Model-based correction (score_mb):
         i. Sample N candidate clean sequences conditioned on (a^τ, τ)
        ii. Roll out each candidate via F/R/Q to get return G
       iii. Softmax-weight candidates by normalized G, get weighted mean action sequence ā
        iv. score_mb = (-a^τ + sqrt(ᾱ_τ)*ā) / (1-ᾱ_τ)

       Q-induced guidance (score_mf):
         branch qgrad_first:
           - compute ∇_{a^0_t} Q(z_0, a^0_t) at first action slot only
         branch gradG_elite:
           - keep topK candidates by G
           - compute full-sequence gradient ∇_{a^0_{t:t+H-1}} G
           - aggregate elite gradients into one sequence-level direction
         convert to diffusion-time score scaling (divide by sqrt(ᾱ_τ) where used)

       Unified target:
         score_target = β * score_mf + (1-β) * score_mb

 14) If epsilon-parameterization is used:
       ε_target = - score_target * sqrt(1-ᾱ_τ)

 15) Loss:
       L_diff = ||prediction - target||^2
       where prediction=ε_hat and target=ε_target (or ε for standard denoising samples)

 16) Backprop L_diff; optimizer step on diffusion policy network

E) Target network and logging
 17) Soft-update target Q parameters: θ̄ <- (1-τ_Q)θ̄ + τ_Q θ
 18) Log metrics:
       consistency_loss, reward_loss, value_loss, grad_norm,
       diffusion_loss, score norms, guidance gradient norms, etc.
```

## Config branches
- `policy_update_type`: `gaussian` or `diffusion`.
- `planner_type` + `diffusion_policy_act`: determines which action-generator branch is used at interaction time (relevant for which trajectories feed policy-data storage).
- Unified-score controls:
  - Mixing weight \(\beta\): `diffusion_teacher_beta` (name in config; used as mf/mb mixing coefficient).
  - Guidance mode: `diffusion_mf_mode` in `{qgrad_first, gradG_elite}`.
  - Q usage: `diffusion_mf_q_type`, `diffusion_mf_use_target_q`.

## Compute budget controls
- Only a fraction of each diffusion-policy minibatch uses the expensive unified-target construction (`diffusion_score_distill_batch_frac`); the rest uses standard denoising targets.
- Main expensive terms:
  - Candidate rollouts with latent dynamics/reward/value (`N = diffusion_teacher_num_samples`).
  - Elite gradient computation over top candidates (`topK = diffusion_teacher_topk` or `diffusion_mf_topk` depending on branch).
- Cost scaling (roughly):
  - Rollout cost grows with `N * H`.
  - Full-sequence guidance gradient grows with `topK * H` and autograd through multi-step latent rollouts.
- Practical implication: reducing `N`, `topK`, or the expensive-batch fraction directly lowers update-time latency and memory.

## Paper alignment note
The implementation trains a diffusion action-sequence policy by matching a **unified score field** composed of two components: a model-based correction term from latent rollouts and a Q-structured guidance term from value gradients. The model contributes an implicit correction signal over denoising steps, while critic-induced gradients shape high-value action structure. These are merged via a single mixing coefficient \(\beta\) and then mapped to the epsilon target for MSE training when epsilon parameterization is used.

## Code pointers
- Update entrypoint: `tdmpc2/tdmpc2.py`
- World-model and critic modules: `tdmpc2/common/world_model.py`
- Replay sampling format: `tdmpc2/common/buffer.py`
- Diffusion policy network, noise schedule, q_sample: `tdmpc2/policies/diffusion_policy.py`
- Unified score construction + model-based / Q-guided branches: `tdmpc2/planners/diffusion_planner.py`

## VERIFY
- The config key carrying \(\beta\) currently appears with a legacy name; confirm intended value flow in planner score-mixing code when changing defaults.
