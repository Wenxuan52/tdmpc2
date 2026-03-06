# TD-MPC2 Code Map (Updated for Diffusion Step1/2 + Step3.1/3.2)

> 目的：给后续 **Step4（联合训练 & 用 diffusion policy 替换 `update_pi`）** 提供可直接执行的代码地图、数据流、约束与改造顺序。

---

## A) 运行入口与主流程

### 1) 训练/评估入口
- 训练入口：`tdmpc2/train.py::train(cfg)`。
  - 通过 `parse_cfg` 展开任务配置。
  - 构建 `{env, agent, buffer, logger}`，并选择：
    - `OnlineTrainer`（单任务在线）
    - `OfflineTrainer`（mt30/mt80 离线）
- 评估入口：`tdmpc2/evaluate.py::evaluate(cfg)`。

### 2) Trainer 调用链（关键）
- `OnlineTrainer.train`：
  1. `action = agent.act(...)`
  2. `agent.maybe_store_distill(...)`（若开启 distill）
  3. 环境 step
  4. `agent.update(buffer)`（世界模型 + Q + policy 更新 + 可选 diffusion distill 更新）
- `OfflineTrainer.eval` 也会调用 `agent.act(...)` 与 `agent.maybe_store_distill(task=...)`，可为多任务 teacher 采样提供轨迹。

---

## B) `TDMPC2` 当前结构（Step3 后）

**文件：** `tdmpc2/tdmpc2.py`

### 1) 当前并存的两套“策略更新”
- 旧主线（TD-MPC2 原生）：`update_pi(zs, task)`
  - 使用 `model.pi` 采样动作，Q 加权构造 `pi_loss`，并由 `self.pi_optim` 更新 `_pi`。
- 新增蒸馏支线：`_update_diffusion_policy()`
  - 从 `DistillBuffer` 取 `(z0, a0_plan, task, action_mask)`。
  - 走 DDPM 前向加噪得到 `a_tau`。
  - 训练 `DiffusionPolicy`（epsilon-net）预测噪声。
  - Step3.2 可选启用 teacher score distill（见后文）。

### 2) 当前 `update()` 的实际行为
`update()` 目前顺序：
1. `self._update(...)`：
   - 更新世界模型（encoder/dynamics/reward/termination/Q）
   - 调用 `update_pi(...)` 更新高斯 prior `_pi`
2. 若 `diffusion_distill_enabled=true`：再调用 `_update_diffusion_policy()`。

> 这意味着当前是“**双策略并存**”状态。Step4 的核心就是把第 1 步里的 `update_pi` 退场/替换。

### 3) 规划器路由
- `planner_type=mppi` -> `_plan`（原 MPPI）。
- `planner_type=diffusion` -> `DiffusionPlanner.plan`（Step1/2 teacher planner）。
- `planner_type=diffusion_policy` 且 `diffusion_policy_act=true` -> `_diffusion_policy_plan`（学生网络采样动作序列并取第一个动作）。

---

## C) Diffusion 相关模块职责

### 1) Teacher：`DiffusionPlanner`
**文件：** `tdmpc2/planners/diffusion_planner.py`

- `plan(...)`：在线规划（Step1 + 可选 Step2A/2B），输出执行动作。
- `_estimate_value_grad_elite(...)`：Step2B 的可导整段回报 `G` 估计。
- `_teacher_score_single(...)` / `compute_teacher_scores(...)`（Step3.2）：
  - 输入 `(z0, x_tau, tau, task, mask)`；
  - 先采样 `a0_samples` 并用 `_estimate_value` 算 `G_all`（no-grad）；
  - 得 `score_mb`；
  - 对 topK elite 用可导 rollout + `autograd.grad` 得 `score_mf`；
  - 混合：`score_target = beta*score_mf + (1-beta)*score_mb`。

### 2) Student：`DiffusionPolicy`
**文件：** `tdmpc2/policies/diffusion_policy.py`

- `forward(z, a_tau, tau, task)`：预测 `eps_hat`。
- `q_sample(a0, tau, eps)`：前向加噪。
- `sample(...)`：反向采样动作序列（用于 `planner_type=diffusion_policy`）。

### 3) Distill 数据：`DistillBuffer`
**文件：** `tdmpc2/common/distill_buffer.py`

- 轻量 ring buffer，存 `(z0, a0_plan, task_idx, action_mask)`。
- 来源：`agent.act()` 后记录的 `agent._last_z` + `agent._last_plan` 经 `maybe_store_distill(...)` 写入。

---

## D) Step3.2 当前实现要点（已落地）

在 `_update_diffusion_policy()` 中：
1. 采样 batch（`B=diffusion_distill_batch_size`）。
2. 采样 `tau~Uniform{1..T-1}` 与 `eps~N(0,I)`，构建 `a_tau=q_sample(a0,tau,eps)`。
3. 学生预测 `eps_hat`。
4. 若 `diffusion_score_distill_enabled=true`：
   - 按 `diffusion_score_distill_batch_frac` 抽子集 `B_teacher`；
   - 调用 `compute_teacher_scores` 得 `score_target`；
   - 转换为 `eps_target_teacher = -score_target*sqrt(1-alpha_bar[tau])`；
   - teacher 子集用 score-target MSE，其余样本继续用标准 Step3.1 的 injected-eps MSE；
   - 加权合并 loss。
5. 仅 `diffusion_policy_optim.step()` 更新学生网络。

### 当前日志（可用于验活）
- 基础：
  - `diffusion_policy/loss`
  - `diffusion_policy/loss_standard`
  - `diffusion_policy/loss_teacher`
  - `diffusion_policy/eps_norm`
  - `diffusion_policy/eps_hat_norm`
- Step3.2 teacher 统计：
  - `diffusion_policy/teacher_score_norm`
  - `diffusion_policy/teacher_score_mb_norm`
  - `diffusion_policy/teacher_score_mf_norm`
  - `diffusion_policy/teacher_grad_norm`
  - `diffusion_policy/teacher_eps_target_norm`

---

## E) Step4 任务规划（联合训练 & 替换 `update_pi`）

下面是基于当前代码框架、可最小侵入执行的改造计划。

### 目标
把“策略更新”从高斯 prior `_pi` 的 `update_pi`，迁移为 diffusion student 的 `update_diffusion_policy`（以 Step3 的 score-matching 为主），同时保持世界模型 / Q 的 TD-MPC2 主线训练不变。

### 关键原则
1. **世界模型/Q 主线不动**：仍在 `_update(...)` 中按现有 TD 目标更新。
2. **策略更新切换**：`update_pi` 从主训练路径退出；由 `update_diffusion_policy` 接管 actor update 职责。
3. **Step2B 约束保留**：`score_mf` 必须是整段动作序列 ∇G（elite 可导 rollout）。
4. **无梯度污染**：
   - teacher score 构造只对 action leaf 做 `autograd.grad`；
   - 不调用 teacher 的 `.backward()`；
   - 优化器 step 仅针对目标策略参数（学生网络）。

### 推荐改造顺序

#### Phase 1：引入策略更新开关（兼容迁移）
新增例如：
- `policy_update_type: gaussian | diffusion`
- 默认 `gaussian`（保证回归）

在 `TDMPC2.update()` 内：
- `gaussian`：保留现状（`update_pi` + 可选 distill）
- `diffusion`：跳过 `update_pi`，仅做 diffusion policy 更新（可用 Step3.1/3.2）

> 这样可 A/B 对照，降低一次性切换风险。

#### Phase 2：将 diffusion 更新“并入主 actor 更新语义”
- 让 `_update(...)` 的返回统计中包含统一 actor 指标（例如 `actor_loss` 映射到 diffusion loss）。
- 统一 logger 命名，避免训练脚本上层逻辑依赖 `pi_*` 时丢字段。

#### Phase 3：planner 退火到 student 主导
建议在配置中增加调度参数（按 step 线性或分段）：
- `student_act_prob`：训练时按概率用 `diffusion_policy` 直接 act；否则走 teacher planner。
- `teacher_beta_schedule`：`diffusion_teacher_beta` 随训练升高（更信 mf / Q 结构）。
- `teacher_num_samples_schedule`：`diffusion_teacher_num_samples` 逐步下降。

预期阶段：
1. 早期：teacher planner 主导，student 主要学习 teacher field。
2. 中期：混合执行（teacher + student）。
3. 后期：student 直接采样为主，teacher 仅低频纠偏或验证。

### Step4 代码触点（必须优先阅读）
1. `tdmpc2/tdmpc2.py`
   - `_update`, `update`, `update_pi`, `_update_diffusion_policy`, `act`, `plan`。
2. `tdmpc2/planners/diffusion_planner.py`
   - `_estimate_value_grad_elite`, `compute_teacher_scores`, `plan`。
3. `tdmpc2/policies/diffusion_policy.py`
   - `forward`, `q_sample`, `sample`。
4. `tdmpc2/trainer/online_trainer.py` 与 `offline_trainer.py`
   - `maybe_store_distill` 的调用时机。

---

## F) 风险点与检查清单（Step4 前必须明确）

### 1) 梯度隔离
- teacher score 构造阶段检查 `model.parameters()` 的 `.grad` 不应被写入。
- 学生更新前后检查仅 `diffusion_policy` 参数梯度非空。

### 2) 训练吞吐
- Step3.2 开销主要来自 `compute_teacher_scores`：
  - 重点调小 `diffusion_teacher_num_samples`, `diffusion_teacher_topk`, `diffusion_score_distill_batch_frac`。

### 3) 多任务一致性
- `task_batch` 和 `action_mask` 在 teacher/student 两条分支都要一致应用。
- 统计中按任务分桶查看 `teacher_score_norm` 量级，避免某些任务退化为零。

### 4) 行为兼容
- 默认配置必须仍是原行为：
  - `planner_type=mppi|diffusion` 不变；
  - Step3.2 默认关闭；
  - Step4 开关默认不切换到 diffusion actor update。

---

## G) 配置分组（当前已存在）

### Distill 基础（Step3.1）
- `diffusion_distill_enabled`
- `diffusion_distill_buffer_size`
- `diffusion_distill_batch_size`
- `diffusion_policy_lr`
- `diffusion_policy_weight_decay`
- `diffusion_policy_steps`
- `diffusion_policy_act`

### Score-distill（Step3.2）
- `diffusion_score_distill_enabled`
- `diffusion_teacher_num_samples`
- `diffusion_teacher_topk`
- `diffusion_teacher_temp`
- `diffusion_teacher_eta`
- `diffusion_teacher_beta`
- `diffusion_teacher_std_floor`
- `diffusion_teacher_detach_weights`
- `diffusion_teacher_grad_norm_clip`
- `diffusion_score_distill_batch_frac`

---

## H) 运行建议（面向 Step4 预研）

### 小算力首跑（dog-run）
- `planner_type=diffusion`
- `diffusion_steps=10`
- `diffusion_distill_enabled=true`
- `diffusion_score_distill_enabled=true`
- `diffusion_teacher_num_samples=32`
- `diffusion_teacher_topk=8~16`
- `diffusion_score_distill_batch_frac=0.1~0.25`

### 验证 teacher target 非零
重点看：
- `diffusion_policy/teacher_score_norm`
- `diffusion_policy/teacher_eps_target_norm`
- `diffusion_policy/teacher_grad_norm`

若这些长期接近 0：优先排查 mask、topk 太小、`beta` 太低或 `std_floor/temp` 设置。
