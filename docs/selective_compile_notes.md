# Selective `torch.compile` Notes

This repository now uses a **minimal selective compile path**.

## What is compiled

- `TDMPC2._compute_world_model_losses` is compiled when `cfg.compile: true`.
- Compile invocation is conservative:
  - `torch.compile(..., fullgraph=False, dynamic=False)`

## What is explicitly kept eager

The following functions are explicitly marked with `@torch.compiler.disable`:

- `TDMPC2.act`
- `TDMPC2._plan`
- `TDMPC2._estimate_value`
- `DiffusionPlanner.plan`
- `DiffusionPlanner.compute_teacher_scores`
- `DiffusionPlanner._teacher_score_single`
- `DiffusionPlanner._estimate_value_grad_elite`
- `OnlineTrainer.eval`
- `OfflineTrainer.eval`
- `evaluate.evaluate`

## How to enable selective compile

- Set `compile: true` in config (or via CLI override).
- With compile disabled (`compile: false`, default), behavior follows the eager path.

## Intentionally unchanged

- Training algorithm and semantics.
- Planner, diffusion, and teacher-score logic.
- Evaluation behavior.
- Replay/buffer logic and environment interaction.
- Default hyperparameters.
