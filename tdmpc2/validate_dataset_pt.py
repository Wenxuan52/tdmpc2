import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
from tensordict import TensorDict

DEFAULT_REPLAY_ROOT = Path('/media/datasets/cheliu21/cxy_worldmodel/replay')
TASKS_TO_VALIDATE = [
	"dog-run",
	"dog-walk",
	"dog-trot",
	"dog-stand",
]


def _print_value_stats(name, value):
	print(f'{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}')
	if value.numel() == 0:
		return
	if torch.is_floating_point(value):
		finite = torch.isfinite(value)
		large = finite & (value.abs() > 1e6)
		print(
			f'  finite={finite.all().item()} nan_count={torch.isnan(value).sum().item()} '
			f'large_finite_count={large.sum().item()}'
		)


def _print_episode_summary(td: TensorDict):
	first = td[0]
	print('[first episode]')
	print(f'  batch_size={first.batch_size}')
	print(f'  keys={list(first.keys())}')
	for key in first.keys():
		value = first[key]
		if hasattr(value, 'shape'):
			print(f'  {key}: shape={tuple(value.shape)}, dtype={value.dtype}')
		else:
			print(f'  {key}: type={type(value)}')


def validate_dataset_pt(fp: str):
	td = torch.load(fp, weights_only=False)
	print('=' * 80)
	print(f'file: {fp}')
	print(f'type: {type(td)}')
	if not isinstance(td, TensorDict):
		raise TypeError(f'Expected TensorDict, got {type(td)}.')
	print(f'batch_size: {td.batch_size}')
	print(f'keys: {list(td.keys())}')

	for key in td.keys():
		value = td[key]
		if hasattr(value, 'shape'):
			_print_value_stats(key, value)
		else:
			print(f'{key}: type={type(value)}')

	if 'task' in td.keys():
		task = td['task']
		task_constant = bool((task == task[:, :1]).all().item())
		task_ids, counts = torch.unique(task[:, 0], return_counts=True)
		print(f'task_constant_within_episode: {task_constant}')
		print(f'task_ids: {task_ids.tolist()}')
		print(f'task_episode_counts: {counts.tolist()}')

	if 'reward' in td.keys():
		print(f'reward[:, 0] nan_count: {torch.isnan(td["reward"][:, 0]).sum().item()}')
	if 'action' in td.keys():
		action0 = td['action'][:, 0]
		print(f'action[:, 0] nan_count: {torch.isnan(action0).sum().item()}')
		finite = torch.isfinite(td['action'])
		large = finite & (td['action'].abs() > 1e6)
		print(f'action abs>1e6 count: {large.sum().item()}')
	if 'terminated' in td.keys():
		print(f'terminated[:, 0] nan_count: {torch.isnan(td["terminated"][:, 0]).sum().item()}')

	_print_episode_summary(td)
	print('=' * 80)


def _shape_signature_tensordict(td: TensorDict) -> str:
	obs_shape = tuple(td['obs'].shape) if 'obs' in td.keys() else None
	action_shape = tuple(td['action'].shape) if 'action' in td.keys() else None
	reward_shape = tuple(td['reward'].shape) if 'reward' in td.keys() else None
	return f'TensorDict(batch_size={tuple(td.batch_size)}, obs={obs_shape}, action={action_shape}, reward={reward_shape})'


def _validate_tensordict(td: TensorDict, task: str) -> str:
	if len(td.batch_size) != 2:
		raise ValueError(f'{task}: invalid batch_size={td.batch_size}, expected rank-2 [num_episodes, episode_length].')
	required = {'obs', 'action', 'reward'}
	missing = sorted(required - set(td.keys()))
	if missing:
		raise KeyError(f'{task}: missing required keys {missing}.')
	return _shape_signature_tensordict(td)


def _validate_variable_length_dict(obj: dict, task: str) -> str:
	if not bool(obj.get('variable_length', False)):
		raise ValueError(f'{task}: dict payload missing variable_length=True marker.')
	chunks = obj.get('chunks_by_length')
	if not isinstance(chunks, dict) or not chunks:
		raise ValueError(f'{task}: invalid or empty chunks_by_length.')
	length_sigs = []
	for length_str, td in chunks.items():
		if not isinstance(td, TensorDict):
			raise TypeError(f'{task}: chunks_by_length[{length_str}] is {type(td)}, expected TensorDict.')
		length_sigs.append(f'len={length_str}:{_validate_tensordict(td, task)}')
	return 'VariableLength{' + '; '.join(sorted(length_sigs)) + '}'


def validate_task_file(task: str, replay_root: Path) -> str:
	fp = replay_root / f'{task}.pt'
	if not fp.exists():
		raise FileNotFoundError(f'{task}: missing file {fp}')
	obj = torch.load(fp, weights_only=False)
	if isinstance(obj, TensorDict):
		return _validate_tensordict(obj, task)
	elif isinstance(obj, dict):
		return _validate_variable_length_dict(obj, task)
	else:
		raise TypeError(f'{task}: unexpected top-level type {type(obj)}.')


def main():
	parser = argparse.ArgumentParser(description='Validate replay `.pt` files and print only failures.')
	parser.add_argument(
		'--replay-root',
		type=Path,
		default=DEFAULT_REPLAY_ROOT,
		help='Replay directory containing task-level `.pt` files.',
	)
	args = parser.parse_args()
	shape_counter = Counter()
	shape_to_tasks = defaultdict(list)
	errors = []
	for task in TASKS_TO_VALIDATE:
		try:
			shape_sig = validate_task_file(task, args.replay_root)
			shape_counter[shape_sig] += 1
			shape_to_tasks[shape_sig].append(task)
		except Exception as exc:
			errors.append(f'{task}: {exc}')

	for msg in errors:
		print(msg)

	print('=== shape counts ===')
	for shape_sig, count in shape_counter.most_common():
		print(f'count={count} | {shape_sig}')

	print('=== minority shapes ===')
	if shape_counter:
		minority_threshold = 2
		for shape_sig, count in shape_counter.most_common():
			if count <= minority_threshold:
				tasks = ', '.join(shape_to_tasks[shape_sig])
				print(f'count={count} | tasks=[{tasks}] | {shape_sig}')


if __name__ == '__main__':
	main()
