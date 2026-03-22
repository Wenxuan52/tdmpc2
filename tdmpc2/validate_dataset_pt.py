import argparse

import torch
from tensordict import TensorDict


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


def main():
	parser = argparse.ArgumentParser(description='Validate a TD-MPC2 offline dataset `.pt` file.')
	parser.add_argument('file', help='Path to the dataset `.pt` file.')
	args = parser.parse_args()
	validate_dataset_pt(args.file)


if __name__ == '__main__':
	main()
