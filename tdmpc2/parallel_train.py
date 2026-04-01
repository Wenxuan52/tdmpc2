from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf

from tdmpc2.common import TASK_SET
from tdmpc2.common.replay_buffer_saver import merge_seed_replay_chunks


def _resolve_tasks(cfg) -> list[str]:
	if cfg.get('tasks'):
		return list(cfg.tasks)
	task_set = str(cfg.get('task_set', 'mt80'))
	if task_set not in TASK_SET:
		raise ValueError(f'Unknown task_set "{task_set}". Available: {list(TASK_SET.keys())}')
	return list(TASK_SET[task_set])


def _resolve_gpu_ids(cfg) -> list[int]:
	if cfg.get('gpu_ids'):
		return [int(gpu_id) for gpu_id in cfg.gpu_ids]
	num_gpus = int(cfg.get('num_gpus', 0) or 0)
	if num_gpus <= 0:
		raise ValueError('Please set either `gpu_ids` or `num_gpus > 0` in parallel_config.yaml.')
	return list(range(num_gpus))


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _steps_for_task(cfg, task: str) -> int:
	if task.startswith('mw-'):
		return int(cfg.get('metaworld_steps', cfg.get('default_steps', cfg.get('steps', 0))) or 0)
	if task in TASK_SET['mt30']:
		return int(cfg.get('dmc_steps', cfg.get('default_steps', cfg.get('steps', 0))) or 0)
	return int(cfg.get('default_steps', cfg.get('steps', 0)) or 0)


def _build_seed_command(cfg, task: str, seed: int, temp_replay_root: Path, reward_csv_root: Path, steps: int) -> list[str]:
	train_entry = str(cfg.get('train_entry', 'tdmpc2/train.py'))
	exp_name = str(cfg.get('exp_name', 'parallel-online-diffusion'))
	cmd = [
		str(cfg.get('python_bin', sys.executable)),
		'-u',
		train_entry,
		f'task={task}',
		f'seed={seed}',
		f'steps={steps}',
		f'planner_type={cfg.get("planner_type", "diffusion")}',
		f'obs={cfg.get("obs", "state")}',
		f'exp_name={exp_name}',
		f'compile={str(bool(cfg.get("compile", False))).lower()}',
		f'compile_mode={cfg.get("compile_mode", "max-autotune-no-cudagraphs")}',
		f'diffusion_eval_compile={str(bool(cfg.get("diffusion_eval_compile", False))).lower()}',
		f'diffusion_eval_compile_mode={cfg.get("diffusion_eval_compile_mode", "reduce-overhead")}',
		f'eval_freq={int(cfg.get("eval_freq", 0) or 0)}',
		f'save_model_every={int(cfg.get("save_model_every", 0) or 0)}',
		f'save_agent={str(bool(cfg.get("save_agent", True))).lower()}',
		f'save_video={str(bool(cfg.get("save_video", False))).lower()}',
		f'enable_wandb={str(bool(cfg.get("enable_wandb", False))).lower()}',
		f'save_replay={str(bool(cfg.get("save_replay", True))).lower()}',
		f'replay_save_dir={temp_replay_root.as_posix()}',
		f'replay_flush_every_episodes={int(cfg.get("replay_flush_every_episodes", 1000))}',
		f'replay_include_terminated={str(bool(cfg.get("replay_include_terminated", True))).lower()}',
		f'save_reward_csv={str(bool(cfg.get("save_reward_csv", True))).lower()}',
		f'reward_csv_dir={reward_csv_root.as_posix()}',
	]
	for override in cfg.get('common_overrides', []) or []:
		cmd.append(str(override))
	return cmd


def _launch_seed_processes(cfg, gpu_id: int, task: str, temp_replay_root: Path, reward_csv_root: Path, log_root: Path):
	repo_root = _repo_root()
	steps = _steps_for_task(cfg, task)
	if steps <= 0:
		raise ValueError(f'Invalid step count {steps} for task "{task}".')
	processes = []
	seed_log_dir = log_root / task
	seed_log_dir.mkdir(parents=True, exist_ok=True)
	for seed in cfg.seeds:
		seed = int(seed)
		cmd = _build_seed_command(cfg, task, seed, temp_replay_root, reward_csv_root, steps)
		log_fp = seed_log_dir / f'seed_{seed}.log'
		env = os.environ.copy()
		env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
		print(f'[gpu {gpu_id}] launching task={task} seed={seed} steps={steps}: {" ".join(cmd)}')
		log_handle = open(log_fp, 'w')
		proc = subprocess.Popen(
			cmd,
			cwd=repo_root,
			env=env,
			stdout=log_handle,
			stderr=subprocess.STDOUT,
		)
		processes.append((seed, proc, log_fp, log_handle))
	return processes


def _wait_seed_processes(processes, gpu_id: int, task: str):
	failed = []
	for seed, proc, log_fp, log_handle in processes:
		return_code = proc.wait()
		log_handle.close()
		if return_code != 0:
			failed.append((seed, return_code, log_fp))
		else:
			print(f'[gpu {gpu_id}] task={task} seed={seed} finished successfully. log={log_fp}')
	return failed


def _merge_task_replay(cfg, task: str, temp_replay_root: Path, final_replay_root: Path):
	seed_dirs = [temp_replay_root / task / f'seed_{int(seed)}' for seed in cfg.seeds]
	output_path = final_replay_root / f'{task}.pt'
	summary = merge_seed_replay_chunks(
		task=task,
		seed_dirs=seed_dirs,
		output_path=output_path,
		cleanup=bool(cfg.get('merge_cleanup_temp', False)),
	)
	print(
		f'[merge] task={summary.task} files={summary.num_files} episodes={summary.num_episodes} '
		f'episode_length={summary.episode_length} output={summary.output_path}'
	)


def _gpu_worker(cfg, gpu_id: int, tasks_queue: queue.Queue, final_replay_root: Path, temp_replay_root: Path, reward_csv_root: Path, log_root: Path, failures: list):
	while True:
		try:
			task = tasks_queue.get_nowait()
		except queue.Empty:
			return
		try:
			start = time.time()
			shutil.rmtree(temp_replay_root / task, ignore_errors=True)
			processes = _launch_seed_processes(cfg, gpu_id, task, temp_replay_root, reward_csv_root, log_root)
			failed = _wait_seed_processes(processes, gpu_id, task)
			if failed:
				failures.append((gpu_id, task, failed))
				print(f'[gpu {gpu_id}] task={task} failed: {failed}')
				return
			if bool(cfg.get('save_replay', True)):
				_merge_task_replay(cfg, task, temp_replay_root, final_replay_root)
			elapsed = time.time() - start
			print(f'[gpu {gpu_id}] task={task} completed in {elapsed/60:.1f} minutes.')
		finally:
			tasks_queue.task_done()


def main():
	parser = argparse.ArgumentParser(description='Launch parallel single-task online TD-MPC2 training jobs.')
	parser.add_argument(
		'--config',
		default=str(Path(__file__).resolve().with_name('parallel_config.yaml')),
		help='Path to parallel training YAML config.',
	)
	parser.add_argument('--replay_save_dir', default=None, help='Override replay_save_dir from config.')
	parser.add_argument('--num_gpus', type=int, default=None, help='Override num_gpus from config.')
	parser.add_argument('--tasks', nargs='*', default=None, help='Optional explicit task list override.')
	args = parser.parse_args()

	cfg = OmegaConf.load(args.config)
	if args.replay_save_dir is not None:
		cfg.replay_save_dir = args.replay_save_dir
	if args.num_gpus is not None:
		cfg.num_gpus = args.num_gpus
		cfg.gpu_ids = []
	if args.tasks:
		cfg.tasks = args.tasks
	tasks = _resolve_tasks(cfg)
	gpu_ids = _resolve_gpu_ids(cfg)
	repo_root = _repo_root()
	final_replay_root = (repo_root / Path(str(cfg.get('replay_save_dir', './replays')))).resolve()
	temp_replay_root = final_replay_root / '_tmp'
	reward_csv_root = final_replay_root / '_csv'
	log_root = final_replay_root / '_logs'
	final_replay_root.mkdir(parents=True, exist_ok=True)
	temp_replay_root.mkdir(parents=True, exist_ok=True)
	reward_csv_root.mkdir(parents=True, exist_ok=True)
	log_root.mkdir(parents=True, exist_ok=True)

	print(f'Using GPUs: {gpu_ids}')
	print(f'Tasks to run ({len(tasks)}): {tasks}')
	print(f'Per-task seeds: {[int(seed) for seed in cfg.seeds]}')
	print(f'Final replay dir: {final_replay_root}')
	print(f'Temporary replay dir: {temp_replay_root}')
	print(f'Reward CSV dir: {reward_csv_root}')

	tasks_queue = queue.Queue()
	for task in tasks:
		tasks_queue.put(task)

	failures = []
	threads = []
	for gpu_id in gpu_ids:
		thread = threading.Thread(
			target=_gpu_worker,
			args=(cfg, gpu_id, tasks_queue, final_replay_root, temp_replay_root, reward_csv_root, log_root, failures),
			daemon=False,
		)
		thread.start()
		threads.append(thread)

	for thread in threads:
		thread.join()

	if failures:
		print('Parallel training finished with failures:')
		for gpu_id, task, failed in failures:
			print(f'  gpu={gpu_id} task={task} failures={failed}')
		raise SystemExit(1)

	print('Parallel training completed successfully.')


if __name__ == '__main__':
	main()
