# MT80 多模型并发打包训练（8 GPU）

新增脚本：`nautilus/launch_mt80_multisize_pack.sh`。

目标：在同一批 8 卡资源内，**同时训练 1M / 5M / 19M / 48M**，并通过同卡多进程打包提高小模型 GPU 利用率。

> 为避免“每个进程完整加载一份 100GB 级别 MT80 数据”的内存放大，建议同时启用 `offline_data_mode=mmap`（本仓库已支持），让离线数据以 mmap/lazy 方式按需采样。

## 默认打包策略

每张 GPU 默认并发 7 个作业：

- 1M: 3 个副本
- 5M: 2 个副本
- 19M: 1 个副本
- 48M: 1 个副本

这样可以把低利用率小模型“叠加”起来，通常比单进程训练显著提高 `nvidia-smi` 上的利用率。

## 启动方式

```bash
bash nautilus/launch_mt80_multisize_pack.sh
```

或覆盖关键参数：

```bash
NUM_GPUS=8 \
GPU_IDS="0 1 2 3 4 5 6 7" \
REPLICA_1M=4 REPLICA_5M=2 REPLICA_19M=1 REPLICA_48M=1 \
OMP_THREADS=2 CPUS_PER_GPU=19 \
STAGGER_SECONDS=8 ENABLE_MPS=1 \
bash nautilus/launch_mt80_multisize_pack.sh
```

脚本默认会传 `offline_data_mode=mmap` 给 `train.py`。如需回退旧行为，可设置：

```bash
OFFLINE_DATA_MODE=in_memory bash nautilus/launch_mt80_multisize_pack.sh
```

## 调参建议（把利用率调到 80%~90%）

1. **先保守启动**：`3/2/1/1`（1M/5M/19M/48M）。
2. 观察 10~20 分钟 `nvidia-smi`：
   - 若平均低于 75%，先把 `REPLICA_1M` 从 3 提到 4；
   - 仍偏低，再把 `REPLICA_5M` 从 2 提到 3。
3. 若显存压力或上下文切换导致抖动：
   - 降低 `REPLICA_1M`；
   - 保持 `OMP_THREADS=1~2`；
   - 保留 `ENABLE_MPS=1`。
4. 若 CPU 成为瓶颈（util 上不去但 GPU mem 已高）：
   - 提高 `CPUS_PER_GPU`；
   - 或减少每卡并发作业数。

## 关键实现点

- 同卡多进程 + `CUDA_VISIBLE_DEVICES` 固定到单卡。
- `taskset` 对每个子作业做 CPU 子区间绑定。
- 默认开启 CUDA MPS（可关闭 `ENABLE_MPS=0`）。
- 默认 `offline_data_mode=mmap`，减小多进程离线数据常驻内存。
- 每个子作业独立 `seed` 与输出目录，便于对比和排错。

## 日志与输出

- 日志：`<DEST_ROOT>/<RUN_TAG>/logs/`
- 模型输出：`<DEST_ROOT>/<RUN_TAG>/gpu{ID}/m{size}/rep{idx}/`
