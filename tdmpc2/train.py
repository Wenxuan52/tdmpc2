import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import torch

import hydra
from termcolor import colored

# Profiling
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def build_profiler(profile_dir: Path):
	profile_dir.mkdir(parents=True, exist_ok=True)

	activities = [ProfilerActivity.CPU]
	if torch.cuda.is_available():
		activities.append(ProfilerActivity.CUDA)

	prof = profile(
		activities=activities,
		schedule=schedule(wait=5, warmup=5, active=10, repeat=1),
		on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
		record_shapes=True,
		profile_memory=True,
		with_stack=True,
	)
	return prof

@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)

	if cfg.profiling:
		profile_dir = Path(cfg.work_dir) / "profile"
		prof = build_profiler(profile_dir)

		with prof:
			trainer.train(profiler=prof)

		sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"

		with open(profile_dir / "top20_ops.txt", "w", encoding="utf-8") as f:
			f.write("=== Top 20 ops ===\n")
			f.write(prof.key_averages().table(sort_by=sort_key, row_limit=20))
			f.write("\n\n=== Top 20 ops (group_by_input_shape=True) ===\n")
			f.write(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=20))
			f.write("\n\n=== Top 20 memory ops ===\n")
			f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))

		try:
			metric = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
			prof.export_stacks(str(profile_dir / "stacks.txt"), metric=metric)
		except Exception as e:
			print(f"[Profiler] export_stacks failed: {e}")

		print(colored('Profiler output:', 'yellow', attrs=['bold']), profile_dir)
		print('\nTraining completed successfully')
	else:
		trainer.train()
		print('\nTraining completed successfully')


if __name__ == '__main__':
	import wandb

	
	
	train()
