import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
import warnings
warnings.filterwarnings('ignore')

from dataclasses import asdict, is_dataclass
from multiprocessing import get_context
from types import SimpleNamespace

import hydra
import numpy as np
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed


class ConfigNamespace(SimpleNamespace):
	"""Simple mutable config object with DictConfig-like get()."""

	def get(self, key, default=None):
		return getattr(self, key, default)


def _cfg_to_dict(cfg):
	if is_dataclass(cfg):
		return asdict(cfg)
	return dict(cfg)


def _parse_eval_devices(eval_devices):
	if eval_devices is None:
		return ['cuda:0']
	if isinstance(eval_devices, (list, tuple)):
		devices = [str(device).strip() for device in eval_devices if str(device).strip()]
	else:
		text = str(eval_devices).strip()
		if text.startswith('[') and text.endswith(']'):
			text = text[1:-1]
		devices = [part.strip().strip('"\'') for part in text.split(',') if part.strip()]
	if not devices:
		raise ValueError('eval_devices must contain at least one CUDA device, e.g. [cuda:0,cuda:1].')
	for device in devices:
		if not device.startswith('cuda:'):
			raise ValueError(f'Unsupported eval device {device}. Only CUDA devices are supported.')
	return devices


def _cuda_visible_device(device):
	try:
		return str(int(str(device).split(':', 1)[1]))
	except (IndexError, ValueError) as exc:
		raise ValueError(f'Invalid CUDA device specification: {device}') from exc


def _task_score(task_name, episode_reward, episode_success, multitask):
	if not multitask:
		return None
	return episode_success * 100 if task_name.startswith('mw-') else episode_reward / 10


def _aggregate_episode_metric(values, topk, metric_name):
	if topk < 0:
		raise ValueError(f'topk must be >= 0, got {topk}.')
	if not values:
		raise ValueError(f'{metric_name} must contain at least one value.')
	if topk == 0:
		return float(np.mean(values))
	if topk > len(values):
		raise ValueError(
			f'topk={topk} cannot exceed the number of evaluated episodes ({len(values)}).'
		)
	sorted_values = sorted(values, reverse=True)
	return float(np.mean(sorted_values[:topk]))


def _run_worker_eval(cfg, payload, visible_device, torch, make_env, TDMPC2):
	set_seed(payload['seed'])
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(payload['checkpoint'])

	results = []
	for task_idx in payload['task_indices']:
		task_name = cfg.tasks[task_idx] if cfg.multitask else cfg.task
		ep_rewards, ep_successes = [], []
		for episode_idx in range(payload['eval_episodes']):
			if cfg.multitask:
				obs = env.reset(task_idx=task_idx)
			else:
				obs = env.reset()
			done, ep_reward, t = False, 0.0, 0
			while not done:
				action = agent.act(obs, t0=t == 0, eval_mode=True, task=task_idx if cfg.multitask else None)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			print(f'[device cuda:{visible_device}] task={task_name} episode={episode_idx + 1}/{payload["eval_episodes"]} reward={ep_reward:.3f} success={info["success"]:.3f}', flush=True)
		episode_reward = _aggregate_episode_metric(ep_rewards, payload['topk'], 'ep_rewards')
		episode_success = _aggregate_episode_metric(ep_successes, payload['topk'], 'ep_successes')
		results.append({
			'task_idx': task_idx,
			'task': task_name,
			'episode_reward': episode_reward,
			'episode_success': episode_success,
			'score': _task_score(task_name, episode_reward, episode_success, cfg.multitask),
			'device': payload['device'],
		})
	return results


def _evaluate_worker(payload):
	visible_device = _cuda_visible_device(payload['device'])
	os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
	os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')

	import torch
	from tdmpc2.envs import make_env
	from tdmpc2.tdmpc2 import TDMPC2

	torch.backends.cudnn.benchmark = True
	torch.set_float32_matmul_precision('high')
	torch.cuda.set_device(0)

	cfg = ConfigNamespace(**payload['cfg'])
	return _run_worker_eval(cfg, payload, visible_device, torch, make_env, TDMPC2)


@hydra.main(config_name='config', config_path='.')
def evaluate_mppi(cfg: dict):
	"""Evaluate an MPPI-planner TD-MPC2 checkpoint, optionally in parallel across GPUs."""
	parsed_cfg = parse_cfg(cfg)
	cfg_dict = _cfg_to_dict(parsed_cfg)
	eval_devices = _parse_eval_devices(cfg.get('eval_devices', None))

	assert cfg_dict['planner_type'] == 'mppi', \
		'`evaluate_mppi.py` is intended for planner_type=mppi.'
	assert cfg_dict['eval_episodes'] > 0, 'Must evaluate at least 1 episode.'
	assert cfg_dict.get('topk', 0) >= 0, 'topk must be >= 0.'
	if cfg_dict.get('topk', 0) > cfg_dict['eval_episodes']:
		raise ValueError(
			f'topk={cfg_dict["topk"]} cannot exceed eval_episodes={cfg_dict["eval_episodes"]}.'
		)
	assert os.path.exists(cfg_dict['checkpoint']), f'Checkpoint {cfg_dict["checkpoint"]} not found! Must be a valid filepath.'

	set_seed(cfg_dict['seed'])
	print(colored(f'Task: {cfg_dict["task"]}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg_dict.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg_dict["checkpoint"]}', 'blue', attrs=['bold']))
	print(colored(f'Planner type: {cfg_dict["planner_type"]}', 'blue', attrs=['bold']))
	aggregation_mode = 'average' if cfg_dict.get('topk', 0) == 0 else f'top-{cfg_dict["topk"]} mean'
	print(colored(f'Episode reward aggregation: {aggregation_mode}', 'blue', attrs=['bold']))
	print(colored(f'Episode success aggregation: {aggregation_mode}', 'blue', attrs=['bold']))
	print(colored(f'Eval devices: {eval_devices}', 'blue', attrs=['bold']))

	if cfg_dict.get('save_video', False) and len(eval_devices) > 1:
		raise ValueError('save_video=true is only supported with a single eval device in evaluate_mppi.py.')

	if cfg_dict['multitask']:
		print(colored(f'Evaluating agent on {len(cfg_dict["tasks"])} tasks:', 'yellow', attrs=['bold']))
		task_indices = list(range(len(cfg_dict['tasks'])))
	else:
		print(colored(f'Evaluating agent on {cfg_dict["task"]}:', 'yellow', attrs=['bold']))
		task_indices = [0]
		eval_devices = eval_devices[:1]

	shards = [[] for _ in eval_devices]
	for index, task_idx in enumerate(task_indices):
		shards[index % len(eval_devices)].append(task_idx)
	payloads = [
		{
			'cfg': cfg_dict,
			'checkpoint': cfg_dict['checkpoint'],
			'device': device,
			'task_indices': shard,
			'eval_episodes': cfg_dict['eval_episodes'],
			'topk': cfg_dict.get('topk', 0),
			'seed': cfg_dict['seed'] + worker_idx,
		}
		for worker_idx, (device, shard) in enumerate(zip(eval_devices, shards)) if shard
	]

	ctx = get_context('spawn')
	if len(payloads) == 1:
		worker_outputs = [_evaluate_worker(payloads[0])]
	else:
		with ctx.Pool(processes=len(payloads)) as pool:
			worker_outputs = pool.map(_evaluate_worker, payloads)

	flat_results = [result for worker_result in worker_outputs for result in worker_result]
	flat_results.sort(key=lambda item: item['task_idx'])

	scores = []
	for result in flat_results:
		if result['score'] is not None:
			scores.append(result['score'])
		print(colored(
			f'  {result["task"]:<22}'
			f'\tR: {result["episode_reward"]:.01f}  '
			f'\tS: {result["episode_success"]:.02f}  '
			f'\tGPU: {result["device"]}',
			'yellow',
		))

	if cfg_dict['multitask']:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate_mppi()
