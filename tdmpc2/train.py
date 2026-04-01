import os
import datetime
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.distributed as dist

import hydra
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.common.buffer import Buffer
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.trainer.offline_trainer import OfflineTrainer
from tdmpc2.trainer.online_trainer import OnlineTrainer
from tdmpc2.common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def _init_distributed(cfg):
	"""
	Initialize torch.distributed process group for DDP runs.
	Returns True when process group is initialized in this call path.
	"""
	use_ddp = bool(getattr(cfg, 'use_ddp', False))
	cfg.local_rank = int(os.environ.get('LOCAL_RANK', getattr(cfg, 'local_rank', 0) or 0))
	cfg.global_rank = int(os.environ.get('RANK', getattr(cfg, 'global_rank', 0) or 0))
	cfg.world_size = int(os.environ.get('WORLD_SIZE', getattr(cfg, 'world_size', 1) or 1))
	if torch.cuda.is_available():
		cfg.device = str(getattr(cfg, 'device', f'cuda:{cfg.local_rank}') or f'cuda:{cfg.local_rank}')
	else:
		cfg.device = 'cpu'

	should_init = use_ddp or cfg.world_size > 1
	if not should_init:
		return False
	if not dist.is_available():
		raise RuntimeError('torch.distributed is not available, but use_ddp=true was requested.')
	if dist.is_initialized():
		return False

	backend = str(getattr(cfg, 'distributed_backend', 'nccl' if torch.cuda.is_available() else 'gloo'))
	init_method = str(getattr(cfg, 'distributed_init_method', 'env://'))
	timeout_sec = int(getattr(cfg, 'distributed_timeout_sec', 1800) or 1800)
	dist.init_process_group(
		backend=backend,
		init_method=init_method,
		rank=cfg.global_rank,
		world_size=cfg.world_size,
		timeout=datetime.timedelta(seconds=timeout_sec),
	)
	if torch.cuda.is_available():
		torch.cuda.set_device(cfg.local_rank)
	return True


def _cleanup_distributed():
	if dist.is_available() and dist.is_initialized():
		dist.destroy_process_group()


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
	_init_distributed(cfg)
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
	try:
		trainer.train()
	finally:
		_cleanup_distributed()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
