from pathlib import Path

from common.replay_buffer_saver import merge_seed_replay_chunks

REPLAY_ROOT = Path('/media/datasets/cheliu21/cxy_worldmodel/replay')
TMP_ROOT = REPLAY_ROOT / '_tmp'
TASKS_TO_MERGE = [
	'dog-run',
	'dog-walk',
	'dog-trot',
	'dog-stand',
]


def merge_one_task(task: str):
	task_tmp_dir = TMP_ROOT / task
	if not task_tmp_dir.exists():
		raise FileNotFoundError(f'{task}: missing task tmp dir {task_tmp_dir}')
	seed_dirs = sorted([p for p in task_tmp_dir.iterdir() if p.is_dir() and p.name.startswith('seed_')])
	if not seed_dirs:
		raise FileNotFoundError(f'{task}: no seed_* directories found under {task_tmp_dir}')
	output_path = REPLAY_ROOT / f'{task}.pt'
	summary = merge_seed_replay_chunks(task=task, seed_dirs=seed_dirs, output_path=output_path, cleanup=True)
	print(
		f'[merge] task={summary.task} files={summary.num_files} episodes={summary.num_episodes} '
		f'episode_length={summary.episode_length} output={summary.output_path}'
	)


def main():
	for task in TASKS_TO_MERGE:
		merge_one_task(task)


if __name__ == '__main__':
	main()
