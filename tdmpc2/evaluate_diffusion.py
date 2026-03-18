import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'
import warnings
warnings.filterwarnings('ignore')

import hydra
import numpy as np
import torch
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Evaluate a checkpoint by reusing the same trainer eval path selected by train.py.

	For `mt30` / `mt80`, this follows:
		train.py -> OfflineTrainer -> OfflineTrainer.eval()
	For single-task runs, this follows:
		train.py -> OnlineTrainer -> OnlineTrainer.eval()
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	cfg.enable_wandb = False
	cfg.save_agent = False
	if cfg.save_video:
		print(colored('Warning: trainer-path standalone evaluation disables wandb video logging; save_video will be ignored.', 'red', attrs=['bold']))
		cfg.save_video = False

	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'Planner type: {cfg.get("planner_type", "mppi")}', 'blue', attrs=['bold']))
	print(colored(f'Reuse train.py eval path: True', 'blue', attrs=['bold']))
	print(colored(f'Load dataset before eval: {bool(getattr(cfg, "load_dataset_for_eval", False))}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	env = make_env(cfg)
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	if cfg.multitask and bool(getattr(cfg, 'load_dataset_for_eval', False)):
		trainer._load_dataset()
	trainer.agent.load(cfg.checkpoint)

	results = trainer.eval()
	if cfg.multitask:
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		scores = []
		for task in cfg.tasks:
			reward = results[f'episode_reward+{task}']
			success = results[f'episode_success+{task}']
			scores.append(success * 100 if task.startswith('mw-') else reward / 10)
			print(colored(
				f'  {task:<22}'
				f'\tR: {reward:.01f}  '
				f'\tS: {success:.02f}',
				'yellow'))
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))
	else:
		print(colored(
			f'  {cfg.task:<22}'
			f'\tR: {results["episode_reward"]:.01f}  '
			f'\tS: {results["episode_success"]:.02f}',
			'yellow'))
		print(colored(f'Normalized score: {results["episode_reward"] / 10:.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
