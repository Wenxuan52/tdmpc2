import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint,
	including diffusion planner evaluation with the same eval-time compile path
	used during training.
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'Planner type: {cfg.get("planner_type", "mppi")}', 'blue', attrs=['bold']))
	print(colored(
		f'Diffusion eval compile: {bool(getattr(cfg, "diffusion_eval_compile", False))}',
		'blue', attrs=['bold']))
	print(colored(
		f'Diffusion eval compile cudagraphs: {bool(getattr(cfg, "diffusion_eval_compile_cudagraphs", False))}',
		'blue', attrs=['bold']))
	print(colored(
		f'Diffusion MF forward compile: {bool(getattr(cfg, "diffusion_mf_forward_compile", False))}',
		'blue', attrs=['bold']))
	print(colored(
		f'Diffusion MF forward compile cudagraphs: {bool(getattr(cfg, "diffusion_mf_forward_compile_cudagraphs", False))}',
		'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		eval_task_idx = task_idx if cfg.multitask else None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=eval_task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = agent.act(obs, t0=t == 0, eval_mode=True, task=eval_task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		scores.append(ep_successes * 100 if task.startswith('mw-') else ep_rewards / 10)
		print(colored(
			f'  {task:<22}'
			f'\tR: {ep_rewards:.01f}  '
			f'\tS: {ep_successes:.02f}',
			'yellow'))
	print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
