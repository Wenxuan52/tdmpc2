
### 最大的不确定/风险点（需要提前规避）

1. **框架差异**：MBD 是 JAX + “真实环境 step”，TD-MPC2 是 PyTorch + “latent world model step”。所以不是复制粘贴，而是迁移“采样→加权→score→更新”的数学结构。 ([GitHub][2])
2. **算力/时延**：MBD 默认 `Ndiffuse=100`、`Nsample=2048` 很重。TD-MPC2 原本 MPPI 是 `iterations × num_samples × horizon`，你换 diffusion 后是 `Ndiffuse × Nsample × horizon`，必须把参数降到能实时（例如 Ndiffuse 10~20，Nsample 256~1024）。 ([GitHub][2])
3. **“model-free diffusion”如果按论文式子用 (\nabla_a G) 会引入反传**：TD-MPC2 的 reward/Q 用 two-hot 反解码（`two_hot_inv`），是可导的，但会带来额外开销与数值稳定性挑战。 ([GitHub][3])

---

## 2) 你要实现的“Dream, Correct and Scale”在 TD-MPC2 里怎么对齐？

把论文第 3 节的对象映射到 TD-MPC2：

| 论文对象               | TD-MPC2 里对应                                                        |
| ------------------ | ------------------------------------------------------------------ |
| latent state (z_t) | `model.encode(obs)` 得到的 latent                                     |
| dynamics (F)       | `model.next(z, a)`                                                 |
| reward (R)         | `model.reward(z, a)`（再 `two_hot_inv`）                              |
| terminal / Q       | `model.Q(z, a)`                                                    |
| (G(z, a_{t:t+H}))  | `_estimate_value(z, actions)`（已经是“reward 累积 + 终端 Q”） ([GitHub][1]) |
| 原 TD-MPC2 planner  | `_plan()` 里 MPPI/CEM loop ([GitHub][1])                            |
| 你要替换的 planner      | diffusion reverse loop（参考 MBD） ([GitHub][2])                       |

你的“核心改动点”非常明确：
**把 `TDMPC2._plan()` 中 MPPI 的采样+elite 更新那一段替换成 diffusion 的采样+score 更新。** ([GitHub][1])

---

## 3) 渐进式实现路线（建议按 0→4 步走，每一步都能跑通）

---

### Step 1：只替换 planner：实现 **Model-based diffusion planner（无梯度版）**

这是最稳、最接近 MBD 的第一步：**不引入 (\nabla_a)**，只用“采样→softmax→加权平均→score→更新”。

**做法（最小侵入 TD-MPC2）：**

1. 在 `TDMPC2._plan()` 里，保留前半段：

   * `z = self.model.encode(obs, task)`
   * warm start：用 `self._prev_mean`（TD-MPC2 原本用于 mean shift）作为 diffusion 初始轨迹的“均值/中心”。 ([GitHub][1])
2. 用一套 DDPM schedule（`betas, alphas, alpha_bar`），维护一个“当前带噪轨迹” `x_tau`（形状 `[H, act_dim]`）。
3. 每个 reverse step (\tau)：

   * 从 (q(a^0 | a^\tau)) 采样很多候选 `a0_samples`（形状 `[Nsample, H, act_dim]`），并 `clamp(-1,1)`（MBD 就这么做）。 ([GitHub][2])
   * 把它转成 TD-MPC2 习惯的 `[H, Nsample, act_dim]`，喂给 `_estimate_value(z_repeat, actions)` 得到每条候选的 `G`。 ([GitHub][1])
   * 用 `softmax( temperature * standardized(G) )` 得权重（MBD 会做 mean/std 标准化避免塌缩；TD-MPC2 则用 max-subtraction）。两种都行，建议先抄 MBD 的标准化套路。 ([GitHub][2])
   * 得到加权均值轨迹 `a_bar = Σ w * a0_samples`
   * **用 MBD 的 score 形式**更新到 (\tau-1)：
     `score_mb = 1/(1-alpha_bar[tau]) * (-x_tau + sqrt(alpha_bar[tau]) * a_bar)` ([GitHub][2])
     再按 DDPM 逆过程更新 `x_{tau-1}`（MBD 里也给了类似更新结构）。 ([GitHub][2])
4. 最终输出 `x_0[0]` 作为动作，并把最终轨迹存到 `self._prev_mean` 供下一步 warm-start（对应 receding horizon）。 ([GitHub][1])

**这一阶段你应该观察到：**

* planner 能稳定输出动作（不 NaN、不抖到爆）
* 在少量 step 上能跑通训练（哪怕回报不如 MPPI，也要先能学）

**常见坑（提前加保护）：**

* `G` 的尺度变化很大 → 权重塌缩：照 MBD 做 reward/value 的 `mean/std` 标准化或 log-sum-exp 稳定化。 ([GitHub][2])
* 多任务 action mask：TD-MPC2 在 MPPI 里会 `actions = actions * action_masks`，你 diffusion 采样后也要一样处理。 ([GitHub][1])

---

### Step 2：在 Step 1 基础上加入 **“model-free diffusion”指导（轻量版）**

你说的“model-based diffusion + model-free diffusion”，建议先做**轻量版 guidance**，别一上来就完全按论文式 (7) 做全轨迹梯度期望（很重）。

**推荐两个可渐进的选项：**

**Option A（最省事、效果通常也不错）：Q-gradient guidance（只对第 1 个动作）**

* 在每个 diffusion step，你已经算出了加权均值轨迹 `a_bar`。
* 只对 `a_bar[0]`（第一个动作）打开 `requires_grad=True`，计算 `Q(z, a_bar[0])` 或者 `G` 的近似（比如只看 `Q`），然后取 `∇_{a} Q` 作为 `score_mf` 的方向（类似 classifier guidance）。
* 最后 `score = β*score_mf + (1-β)*score_mb`，β 可以从小到大调。

这一步计算量小，能先把“model-free”信号接进来。

**Option B（更贴论文，但仍可控）：对少量“elite/子集样本”算 (\nabla_a G)**

* 不对全部 Nsample 做反传，只选 topK（比如 32/64 条）样本做 `G` 的梯度，近似论文里 `E[w * ∇G]`。
* TD-MPC2 的 `two_hot_inv` 是可导的（softmax + 加权 bins + symexp），所以 autograd 能跑，但注意数值稳定。 ([GitHub][3])

---

### Step 3：从“纯 planner”过渡到“可学习 diffusion policy”（你真正要的 Dream/Correct/Scale）

到这里，Step 1/2 其实还是“决策时优化”，还没把 diffusion policy **学出来**。

渐进做法是：**先蒸馏（distill）planner → diffusion 网络**，再做联合训练。

**3.1 先蒸馏（让它先能跑得快）**

* 用 Step 1/2 planner 生成的轨迹当作训练数据：`(z_t, a_{t:t+H}^0)`
* 训练一个 `score_net(z, a^τ, τ)`（或者 ε-net）做标准 DDPM 训练：随机采样 τ，加噪得到 `a^τ`，让网络预测 noise/score。
* 这一步先不追求论文的“精确分解”，目标只是：**网络能复现 planner 的输出分布**，把 Nsample 大采样变成小步数小 batch 的网络采样。

**3.2 再加“Correct”：把 world model 当隐式纠偏器**

* 训练时把 `score_target` 写成你 Step 2 里用的那套：
  `score_target = β*score_mf + (1-β)*score_mb`
* 其中 `score_mb` 仍来自 world model rollout 的 Monte Carlo（类似 MBD），`score_mf` 来自 Q（梯度或近似）。 ([GitHub][2])

这样，你就把论文的核心“纠偏结构”落到一个可训练的 score 网络上了。

---

### Step 4：联合训练 & 替换 TD-MPC2 的 `update_pi`

TD-MPC2 现在的 actor 是一个高斯 policy prior `_pi`，更新方式是 `update_pi(zs)` 用 Q 做加权损失。 ([GitHub][1])
当你有 diffusion policy 后，最终要做的是：

* 把 `update_pi` 替换成 `update_diffusion_policy`：

  * 输入：`z_t`（或 `zs`）+ buffer 中的 action sequence / planner 轨迹
  * loss：DDPM denoising loss（外加你定义的 mf/mb guidance 目标）
* planner 逐步从“重采样”退火到“网络直接采样”：

  * 训练早期：多用 mb（模型纠偏）+ 多 sample
  * 训练后期：β 增大（更信 mf/Q），Nsample 降低，甚至只用网络采样

---

## 4) 你提到的两个“trick”，怎么落到代码上？

### Trick 1：MBD 的 loss 是 trajectory loss，你现在要“多学一个 G”

在 TD-MPC2 里 **不需要额外再学一个全新的 G 网络**就能先跑：

* 直接用 `_estimate_value()` 作为 G（它本来就是用 reward model + terminal Q 来算回报的）。 ([GitHub][1])
  等你 Step 3 想提速/稳定，再考虑：
* 额外训练一个 `G_net(z, a_seq)` 逼近 `_estimate_value`（作为快速 evaluator），减少 rollout 次数。

### Trick 2：MPPI 采样不要了，换成 model-based diffusion + model-free diffusion

对应的最小改动就是：

* 把 `_plan()` 里 `for _ in range(self.cfg.iterations): ...` 那段 MPPI loop 整段替换成 diffusion reverse loop；其它保持不动（编码、mask、warm start、返回第一个动作、保存 prev_mean）。 ([GitHub][1])

---

## 5) 我建议你先做的“最小可运行版本”（MVP）长什么样？

如果只选一条最稳的落地路线：

1. **实现 Step 1：无梯度 model-based diffusion planner**
2. `G` 直接用 `_estimate_value`
3. `score_mb` 直接照 MBD 写法（softmax 权重 + 加权平均 + score 更新）
4. Ndiffuse=10~20，Nsample=256~1024，horizon 先用 TD-MPC2 默认
5. 跑一个最小任务（DMControl 里简单的），对比 MPPI 曲线

等这一步曲线能跑起来，再加 Step 2 的 model-free guidance（先从 Option A：只对第一个动作做 Q-gradient 开始）。

---

如果你愿意，我可以基于 `TDMPC2._plan()` 的现有张量形状（`[H, N, act_dim]`）给你一份更“贴代码”的伪代码骨架（函数签名/形状/哪里需要 no_grad、哪里需要 enable_grad、哪里要 action mask），这样你基本照抄就能开工。

[1]: https://raw.githubusercontent.com/nicklashansen/tdmpc2/main/tdmpc2/tdmpc2.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/LeCAR-Lab/model-based-diffusion/main/mbd/planners/mbd_planner.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/nicklashansen/tdmpc2/main/tdmpc2/common/math.py "raw.githubusercontent.com"
