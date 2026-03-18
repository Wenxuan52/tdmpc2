import dataclasses
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from termcolor import colored

from common.parser import cfg_to_dataclass, parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def _parse_eval_devices(eval_devices, fallback_device):
	"""Normalize eval_devices config into a list of device strings."""
	if eval_devices is None:
		return [fallback_device]
	if isinstance(eval_devices, str):
		devices = [d.strip() for d in eval_devices.split(',') if d.strip()]
		return devices or [fallback_device]
	devices = [str(d).strip() for d in eval_devices if str(d).strip()]
	return devices or [fallback_device]



def _split_task_indices(num_tasks, num_shards):
	"""Split task indices as evenly as possible across shards."""
	indices = list(range(num_tasks))
	return [indices[i::num_shards] for i in range(num_shards)]



def _task_score(task, ep_reward, ep_success):
	if task.startswith('mw-'):
		return ep_success * 100
	return ep_reward / 10



def _evaluate_task_indices(cfg, task_indices, prefix=''):
	"""Evaluate selected task indices for a parsed config and return metrics."""
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)

	results = []
	for task_idx in task_indices:
		task = cfg.tasks[task_idx]
		eval_task_idx = task_idx if cfg.multitask else None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=eval_task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				action = agent.act(obs, t0=t == 0, eval_mode=True, task=eval_task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = float(np.mean(ep_rewards))
		ep_successes = float(np.mean(ep_successes))
		result = {
			'task_idx': task_idx,
			'task': task,
			'episode_reward': ep_rewards,
			'episode_success': ep_successes,
		}
		if cfg.multitask:
			result['score'] = _task_score(task, ep_rewards, ep_successes)
		print(colored(
			f'{prefix}  {task:<22}\tR: {ep_rewards:.01f}  \tS: {ep_successes:.02f}',
			'yellow'))
		results.append(result)
	return results



def _worker_main(rank, cfg_dict, device, task_indices, result_queue):
	cfg_dict = dict(cfg_dict)
	cfg_dict['device'] = device
	cfg = cfg_to_dataclass(OmegaConf.create(cfg_dict))
	set_seed(cfg.seed)
	results = _evaluate_task_indices(cfg, task_indices, prefix=f'[worker {rank} | {device}]')
	result_queue.put((rank, results))



def _evaluate_single_process(cfg):
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		task_indices = list(range(len(cfg.tasks)))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
		task_indices = [0]
	results = _evaluate_task_indices(cfg, task_indices)
	if cfg.multitask:
		scores = [r['score'] for r in results]
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))



def _evaluate_multi_gpu(cfg, cfg_dict, devices):
	assert cfg.multitask, 'Multi-device evaluation currently only supports multitask checkpoints.'
	task_splits = [split for split in _split_task_indices(len(cfg.tasks), len(devices)) if split]
	devices = devices[:len(task_splits)]
	print(colored(
		f'Evaluating agent on {len(cfg.tasks)} tasks across {len(devices)} devices: {devices}',
		'yellow', attrs=['bold']))
	for device, task_indices in zip(devices, task_splits):
		task_names = [cfg.tasks[idx] for idx in task_indices]
		print(colored(f'  {device}: {task_names}', 'yellow'))
	ctx = mp.get_context('spawn')
	result_queue = ctx.SimpleQueue()
	processes = []
	for rank, (device, task_indices) in enumerate(zip(devices, task_splits)):
		proc = ctx.Process(target=_worker_main, args=(rank, cfg_dict, device, task_indices, result_queue))
		proc.start()
		processes.append(proc)
	all_results = []
	for _ in processes:
		_, worker_results = result_queue.get()
		all_results.extend(worker_results)
	for proc in processes:
		proc.join()
		if proc.exitcode != 0:
			raise RuntimeError(f'Evaluation worker exited with code {proc.exitcode}.')
	all_results.sort(key=lambda x: x['task_idx'])
	print(colored('Merged evaluation results:', 'yellow', attrs=['bold']))
	for result in all_results:
		print(colored(
			f"  {result['task']:<22}\tR: {result['episode_reward']:.01f}  \tS: {result['episode_success']:.02f}",
			'yellow'))
	scores = [r['score'] for r in all_results]
	print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
		`eval_devices`: optional list/CSV of CUDA devices for task-level multi-GPU evaluation
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
		$ python evaluate.py task=mt80 checkpoint=/path/to/mt80.pt eval_devices=[cuda:0,cuda:1]
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	cfg_dict = dataclasses.asdict(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	devices = _parse_eval_devices(cfg.get('eval_devices', None), cfg.get('device', 'cuda:0'))
	if len(devices) > 1:
		_evaluate_multi_gpu(cfg, cfg_dict, devices)
	else:
		cfg.device = devices[0]
		_evaluate_single_process(cfg)


if __name__ == '__main__':
	evaluate()
