from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description='Launch DIFF metric parallel training jobs.')
	default_cfg = Path(__file__).resolve().with_name('diff.yaml')
	parser.add_argument('--config', default=str(default_cfg), help='Path to diff parallel config yaml.')
	parser.add_argument('--num_gpus', type=int, default=None, help='Override num_gpus in config.')
	parser.add_argument('--replay_save_dir', default=None, help='Override replay_save_dir in config.')
	parser.add_argument('--tasks', nargs='*', default=None, help='Optional explicit task list override.')
	args, unknown = parser.parse_known_args()

	cmd = [
		sys.executable,
		'-u',
		str(Path(__file__).resolve().with_name('parallel_train.py')),
		'--config',
		args.config,
	]
	if args.num_gpus is not None:
		cmd += ['--num_gpus', str(args.num_gpus)]
	if args.replay_save_dir is not None:
		cmd += ['--replay_save_dir', str(args.replay_save_dir)]
	if args.tasks:
		cmd += ['--tasks', *args.tasks]
	cmd += unknown

	subprocess.run(cmd, check=True)


if __name__ == '__main__':
	main()
