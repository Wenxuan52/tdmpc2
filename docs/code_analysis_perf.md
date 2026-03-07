# TD-MPC2 训练性能代码分析（2026-03 更新）

> 目标：围绕当前 `torch.compile` 报错与训练耗时，梳理训练调用链、Diffusion 关键入口、teacher score 路径、梯度上下文切换、张量形状规则，并给出可执行的编译策略建议。

---

## 0. 结论先行（TL;DR）

1. **训练主链路是 `trainer -> agent.update() -> _update() -> (world model / Q) + actor update`**，其中 world model 更新是稳定的“单大图”，而 diffusion teacher-score 路径包含大量**动态控制流 + `no_grad/enable_grad` 来回切换 + 小图 autograd**，是 compile 最容易抖动的位置。  
2. 你提供的日志里 `GLOBAL_STATE changed: grad_mode` 高频触发重编译，和代码中的 `@torch.no_grad` 包裹大函数、内部再 `with torch.enable_grad()`（尤其 planner / teacher score）**高度一致**。  
3. 当前框架更适合：**继续用 PyTorch，做“分区编译（selective compile）”**，而不是迁移 JAX。JAX 迁移成本极高（TorchRL/TensorDict/现有 world-model API 全栈重写），短期 ROI 很差。  
4. 你遇到的 `cudagraph_trees` deallocate 报错，建议先走：
   - 禁用 cudagraph（或降低 compile 激进程度）
   - 不把含频繁 grad_mode 切换/teacher score 的路径纳入 compile
   - 仅 compile `_update` 中“纯前后向”子图（encoder/dynamics/reward/Q 前向 + loss）

---

## 1. 训练 `update()` 调用链

### 1.1 在线/离线 trainer 到 agent.update

- Online：`OnlineTrainer.train()` 在每步采样后调用 `self.agent.update(self.buffer)`。  
- Offline：`OfflineTrainer.train()` 在每个 iteration 调用 `train_metrics = self.agent.update(self.buffer)`。  

### 1.2 `TDMPC2.update()` 主入口

`TDMPC2.update(buffer)` 逻辑：

1. `buffer.sample()` 返回 `(obs, action, reward, terminated, task)`。
2. `torch.compiler.cudagraph_mark_step_begin()`。
3. 调用 `self._update(obs, action, reward, terminated, **kwargs)` 完成 world model / critic（以及可选 gaussian actor）更新。
4. 若 `policy_update_type == diffusion`：走 `update_diffusion_actor()`（内部是 `_update_diffusion_policy()`）。
5. 否则若开启 distill：额外跑 `_update_diffusion_policy()`。

### 1.3 `_update()` 里 world model / Q update

`_update()` 主要阶段：

1. **target 计算（no_grad）**
   - `next_z = model.encode(obs[1:])`
   - `td_targets = _td_target(next_z, reward, terminated, task)`
2. `self.model.train()`
3. **latent rollout**：`z_t -> next(z_t, a_t)`，累计 consistency loss
4. **预测头前向**：`Q(_zs, action, return_type='all')`、`reward(_zs, action)`、可选 `termination`
5. **loss 聚合**：reward/value/consistency/termination
6. `total_loss.backward()` + grad clip + `optim.step()`
7. 若 `policy_update_type == gaussian`，调用 `update_pi(zs.detach(), task)`
8. `soft_update_target_Q()`

### 1.4 diffusion actor update（student）

当 `policy_update_type=diffusion`：

- `update_diffusion_actor()` -> `_update_diffusion_policy()`。
- `_update_diffusion_policy()` 做的事：
  1. 从 `DistillBuffer.sample(B)` 取 `(z0, a0, task, action_mask)`
  2. 采样 `tau, eps`，构造 `a_tau = q_sample(a0, tau, eps)`
  3. `eps_hat = diffusion_policy(z0, a_tau, tau, task)`
  4. 可选 teacher-score distill：
     - 子批次调用 `DiffusionPlanner.compute_teacher_scores(...)`
     - 转成 `eps_target_teacher = -score * sqrt(1-alpha_bar_tau)`
  5. `loss.backward()` + `diffusion_policy_optim.step()`

---

## 2. DiffusionPlanner / DiffusionPolicy 关键函数入口

## 2.1 DiffusionPlanner

### A) 在线/评估动作入口

- `TDMPC2._diffusion_plan(...)` -> `DiffusionPlanner.plan(agent, obs, ...)`
- `plan()` 是带 `@torch.no_grad()` 的逆扩散规划主循环：
  - 每个 `tau` 采样 `a0_samples`，评估 `values = agent._estimate_value(...)`
  - 计算 model-based score `score_mb`
  - 若 `mf_mode=qgrad_first` 或 `gradG_elite`，在局部 `with torch.enable_grad()` 计算 score_mf
  - 合成 `score = (1-mβ)score_mb + mβ score_mf`，更新 `x_tau`

### B) teacher score 入口（用于蒸馏）

- `compute_teacher_scores(agent, z0_batch, a_tau_batch, tau_batch, ...)`
- 对 batch 内每个样本调用 `_teacher_score_single(...)`

### C) `gradG_elite` 的核心

- `_teacher_score_single()`：
  1. 先 no_grad 评估 `G_all`
  2. `topk(G_all, K)` 选 elite actions
  3. `with torch.enable_grad()`：`a0_elite.requires_grad_(True)`
  4. `G_elite = _estimate_value_grad_elite(...)`
  5. `obj = Σ w_elite * η * G_elite`
  6. `grad_elite = autograd.grad(obj, a0_elite)`
  7. `score_mf = grad_bar / sqrt(alpha_bar_tau)`，再与 `score_mb` 融合

## 2.2 DiffusionPolicy

关键入口三处：

1. `q_sample(a0, tau, eps)`：前向扩散（训练噪声构造）
2. `forward(z, a_tau, tau, task)`：epsilon 网络预测 `eps_hat`
3. `sample(z, task, action_mask, steps)`：`@torch.no_grad()` 逆扩散采样动作轨迹

---

## 3. teacher score 计算路径（含 topK ∇G）

完整路径：

1. `TDMPC2._update_diffusion_policy()`
2. `self._diffusion_planner.compute_teacher_scores(...)`
3. `DiffusionPlanner._teacher_score_single(...)`
4. 内部：
   - no_grad 估计 `G_all`（全样本）
   - 由 `G_all` 得到 `score_mb`
   - `topk(G_all,K)` 选 elite
   - enable_grad 下调用 `_estimate_value_grad_elite(...)`
   - `autograd.grad` 得 `grad_elite`，聚合后得到 `score_mf`
   - `score_target = β*score_mf + (1-β)*score_mb`
5. 返回 `teacher_scores` 给 `_update_diffusion_policy()`，再映射为 `eps_target_teacher`

`_estimate_value_grad_elite(...)` 内部又会调用：

- `agent.model.reward(z, a_t, task)`
- `agent.model.next(z, a_t, task)`
- 最后 `agent.model.pi(z, task)` + `agent.model.Q(z, action, ...)`

所以 topK ∇G 这段本质是“**可微 rollout + terminal Q bootstrap**”的梯度。

---

## 4. `torch.no_grad` 与 `torch.enable_grad` 的现状

## 4.1 no_grad 使用点（关键）

- `TDMPC2.act`（整段无梯度）
- `TDMPC2._estimate_value`
- `TDMPC2._plan`
- `TDMPC2._td_target`
- `TDMPC2._update` 内部 target 计算块：`with torch.no_grad(): next_z + td_targets`
- `DiffusionPlanner.plan`（函数装饰器）
- `DiffusionPlanner._teacher_score_single` 内部 `with torch.no_grad()` 评估 `G_all`
- `DiffusionPolicy.sample`（函数装饰器）

## 4.2 enable_grad 使用点（关键）

- `DiffusionPlanner.plan` 内：
  - `mf_mode=qgrad_first` 时对 `a0` 求 `∇_a Q`
  - `mf_mode=gradG_elite` 时对 elite 动作求 `∇_a G`
- `DiffusionPlanner._teacher_score_single` 内：
  - topK elite 上 `∇_a G`

## 4.3 对 compile 的直接影响

你的日志核心就是：`GLOBAL_STATE changed: grad_mode` 导致重复编译。当前结构是**大 no_grad 函数内部嵌套 enable_grad 小块**，会触发 Dynamo guard 失败；再叠加 `vmap` / `TensorDict` / 动态 shape 与对象状态 guard，重编译概率进一步上升。

---

## 5. tensor shapes 与 batch 维度规则

以下按训练常用维度记号：

- `B`: 训练 batch（config: `batch_size`）
- `H`: horizon（config: `horizon`）
- `A`: action_dim
- `Z`: latent_dim
- `N`: diffusion/planner samples（`diffusion_num_samples` 或 teacher num_samples）
- `K`: top-k elites（`diffusion_mf_topk` / `diffusion_teacher_topk`）

## 5.1 ReplayBuffer 采样后

- `obs`: `[H+1, B, ...obs_dim]`
- `action`: `[H, B, A]`（由 buffer 里 `[1:]` 得到）
- `reward`: `[H, B, 1]`
- `terminated`: `[H, B, 1]`
- `task`（多任务）: `[B]`

## 5.2 `_update()` 内关键形状

- `next_z = encode(obs[1:])`: `[H, B, Z]`
- `z = encode(obs[0])`: `[B, Z]`
- `zs`: `[H+1, B, Z]`
- `_zs = zs[:-1]`: `[H, B, Z]`
- `Q(_zs, action, return_type='all')`: `[num_q, H, B, num_bins]`（若离散 two-hot value）
- `reward_preds`: `[H, B, num_bins]`（或标量回归维）
- `td_targets`: `[H, B, 1]`

## 5.3 DiffusionPolicy 训练形状

- 从 DistillBuffer 取：
  - `z0`: `[B, Z]`
  - `a0`: `[B, H, A]`
  - `task`: `[B]`
  - `action_mask`: `[B, A]`
- `tau`: `[B]`
- `eps`: `[B, H, A]`
- `a_tau = q_sample(...)`: `[B, H, A]`
- `eps_hat = forward(...)`: `[B, H, A]`
- teacher score 子批：`teacher_scores`: `[B_teacher, H, A]`

## 5.4 DiffusionPlanner 规划形状

- `z0 = encode(obs)`: `[1, Z]`（act 时单样本）
- `z = z0.repeat(N,1)`: `[N, Z]`
- `x_tau`: `[H, A]`
- `a0_samples`: `[N, H, A]`
- `actions_for_value = permute -> [H, N, A]`
- `values = _estimate_value(...)`: `[N]`（原始为 `[N,1]` 后 squeeze）
- `a0_elite`: `[K, H, A]`
- `grad_elite`: `[K, H, A]`
- `grad_bar = sum_K`: `[H, A]`
- 最终动作 `action = x0[0]`: `[A]`

## 5.5 batch 维度约定总结

- world model 训练默认“时间在前”：`[H, B, ...]`
- diffusion actor 训练默认“batch 在前”：`[B, H, A]`
- planner 采样临时是“样本在前”：`[N, H, A]`，喂 value 前转为 `[H, N, A]`

这个“多种维度约定并存”是性能优化时要重点关注的（频繁 permute/reshape 会增加图复杂度与内存压力）。

---

## 6. 编译策略建议：JAX 还是 torch.compile？

## 6.1 现状判断

从代码结构看，当前项目是典型 PyTorch 生态深绑定：

- `TensorDict` / `torchrl` replay buffer
- `torch.vmap` + ensemble params 管理
- 训练循环与 planner 深度耦合 PyTorch autograd 语义

**结论：短中期应继续 PyTorch，不建议切到 JAX。**

原因不是 JAX 性能不行，而是迁移成本与风险远高于收益：

- world model + planner + distill + buffer 全链路需重写
- 现有功能（多任务 mask、teacher score 蒸馏、在线/离线 trainer）回归验证成本巨大
- 你当前 bottleneck 首先是 compile 稳定性（重编译/cudagraph），并非 PyTorch 本体上限

## 6.2 推荐的 torch.compile 落地方式（分区编译）

### A) 不要“一把梭” compile 整个 `agent._update`

`_update` 里含 no_grad 块 + train/eval 状态切换 + target Q soft update，图边界复杂。

建议拆成：

- 可编译块：
  - world model 前向（encode/next/reward/Q）+ loss 计算
  - diffusion_policy epsilon_net 前向（纯张量路径）
- 不编译块：
  - `compute_teacher_scores` / `DiffusionPlanner.plan`
  - 所有含 `enable_grad` 小内核、`autograd.grad` 显式调用路径

### B) 对当前报错的针对性措施

你日志出现两类强信号：

1. `GLOBAL_STATE changed: grad_mode`（重编译）
2. cudagraph invariant 报错：input tensor deallocate mismatch

可按优先级尝试：

1. 先禁用 cudagraph（仅保留 inductor）验证稳定性
2. 把含 grad_mode 切换的函数显式排除 compile（planner/teacher score）
3. 减少 Python-side 动态对象 guard（尤其 tensordict memo 动态增长）
4. 若仍不稳，world model 保持 eager + amp，先用 profiler 优化 kernel 热点

### C) 与你给出的 profile 对照

你给的 top op 显示大量 `addmm/mm/layer_norm/mish`，说明主耗时是 MLP 密集算子；这类在 PyTorch eager + AMP 下已经可获得不错吞吐。当前最该先解决的是**编译抖动/失败导致的额外开销**，而不是框架迁移。

---

## 7. 给本次报错的根因定位（结合日志）

你给的 single-task / multi-task 日志都指向同一现象：

- `Q`、`layers.Ensemble.forward(vmap)`、`_process_batched_inputs` 等函数反复 recompile
- 触发 guard 常见是 `GLOBAL_STATE changed: grad_mode`
- 最终在 inductor cudagraph tree invariant 上报 tensor 生命周期不一致

结合代码可以推断：

1. 训练步中既有 `@no_grad` 装饰函数，也有局部 `enable_grad` 的显式反向求导（planner/teacher）。
2. 部分路径里 `TensorDict`/`vmap`/target-detach Q module 共同参与，guard 条件复杂。
3. 在 compile + cudagraph 叠加时，输入张量生命周期与 replay 不完全一致，触发 runtime error。

这不是你配置“错了”，而是当前代码结构对全量 compile 不友好。

---

## 8. 实操优先级（建议按顺序）

1. **先稳**：compile 仅用于 world-model 纯训练图；planner/teacher-score 全部 eager。  
2. **再快**：开启 AMP（如未启用）+ 检查 batch/horizon 对齐，减少小 kernel 与 shape 抖动。  
3. **再精调**：对 diffusion distill 子图做独立 compile 试验（仅 `DiffusionPolicy.forward + loss`），teacher score 仍 eager。  
4. **最后**：若仍需更高吞吐，再考虑更激进 graph capture/自定义 fused kernel；不建议直接迁 JAX。

---

## 9. 一句话建议

对于你当前“训练时间优化 + torch.compile 报错”的阶段，**最佳路线是 PyTorch selective compile（分区编译）+ 稳定 grad_mode 边界**，而不是迁移 JAX。先把重编译和 cudagraph 崩溃压住，收益会比大规模重构更直接。
