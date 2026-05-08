from __future__ import annotations

import argparse
import csv
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf

from common import TASK_SET

DMCONTROL_TASKS = [
    'acrobot-swingup',
    'cartpole-balance',
    'cartpole-balance-sparse',
    'cartpole-swingup',
    'cartpole-swingup-sparse',
    'cheetah-jump',
    'cheetah-run',
    'cheetah-run-back',
    'cheetah-run-backwards',
    'cheetah-run-front',
    'cup-catch',
    'cup-spin',
    'dog-run',
    'dog-trot',
    'dog-stand',
    'dog-walk',
    'finger-spin',
    'finger-turn-easy',
    'finger-turn-hard',
    'fish-swim',
    'hopper-hop',
    'hopper-hop-backwards',
    'hopper-stand',
    'humanoid-run',
    'humanoid-stand',
    'humanoid-walk',
    'pendulum-spin',
    'pendulum-swingup',
    'quadruped-run',
    'quadruped-walk',
    'reacher-easy',
    'reacher-hard',
    'reacher-three-easy',
    'reacher-three-hard',
    'walker-run',
    'walker-run-backwards',
    'walker-stand',
    'walker-walk',
    'walker-walk-backwards',
]
METAWORLD_TASKS = [task for task in TASK_SET['mt80'] if task.startswith('mw-')]
MANISKILL2_TASKS = ['lift-cube', 'pick-cube', 'stack-cube', 'pick-ycb', 'turn-faucet']
MYOSUITE_TASKS_LIST = [
    'myo-reach',
    'myo-reach-hard',
    'myo-pose',
    'myo-pose-hard',
    'myo-obj-hold',
    'myo-obj-hold-hard',
    'myo-key-turn',
    'myo-key-turn-hard',
    'myo-pen-twirl',
    'myo-pen-twirl-hard',
]
VISUAL_DMCONTROL_TASKS = [
    'acrobot-swingup',
    'cheetah-run',
    'finger-spin',
    'finger-turn-easy',
    'finger-turn-hard',
    'quadruped-walk',
    'reacher-easy',
    'reacher-hard',
    'walker-run',
    'walker-walk',
]
DOMAIN_TASKS = {
    'DMControl': DMCONTROL_TASKS,
    'MetaWorld': METAWORLD_TASKS,
    'ManiSkill2': MANISKILL2_TASKS,
    'MyoSuite': MYOSUITE_TASKS_LIST,
    'VisualRL': VISUAL_DMCONTROL_TASKS,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_gpu_ids(cfg) -> list[int]:
    if cfg.get('gpu_ids'):
        return [int(x) for x in cfg.gpu_ids]
    n = int(cfg.get('num_gpus', 0) or 0)
    if n <= 0:
        raise ValueError('num_gpus must be > 0')
    return list(range(n))


def _build_task_queue(tasks: list[str]) -> queue.Queue:
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    return q


def _build_cmd(cfg, task: str, output_root: Path) -> list[str]:
    return [
        str(cfg.get('python_bin', sys.executable)),
        '-u',
        'tdmpc2/timing_worker.py',
        f'task={task}',
        f'planner_type={cfg.get("planner_type", "mppi")}',
        f'obs={cfg.get("obs", "state")}',
        f'compile={str(bool(cfg.get("compile", False))).lower()}',
        f'compile_mode={cfg.get("compile_mode", "reduce-overhead")}',
        f'diffusion_eval_compile={str(bool(cfg.get("diffusion_eval_compile", False))).lower()}',
        f'diffusion_eval_compile_mode={cfg.get("diffusion_eval_compile_mode", "reduce-overhead")}',
        f'+timing_output_dir={output_root.as_posix()}',
        f'+warmup_evals={int(cfg.get("warmup_evals", 2))}',
        f'+measure_evals={int(cfg.get("measure_evals", 10))}',
        f'steps={int(cfg.get("steps", 20000))}',
    ]


def _run_worker(cfg, gpu_id: int, task_queue: queue.Queue, output_root: Path, failures: list[tuple[int, str, int]]):
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            cmd = _build_cmd(cfg, task, output_root)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f'[gpu {gpu_id}] start task={task}')
            rc = subprocess.call(cmd, cwd=_repo_root(), env=env)
            if rc != 0:
                failures.append((gpu_id, task, rc))
                print(f'[gpu {gpu_id}] task={task} failed rc={rc}')
            else:
                print(f'[gpu {gpu_id}] task={task} done')
        finally:
            task_queue.task_done()


def _aggregate_and_save(domain: str, tasks: list[str], output_root: Path):
    training_csv = output_root / f'{domain}_training.csv'
    eval_csv = output_root / f'{domain}_evaluation.csv'
    training_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(training_csv, 'w', newline='') as f_train, open(eval_csv, 'w', newline='') as f_eval:
        tw = csv.writer(f_train)
        ew = csv.writer(f_eval)
        tw.writerow(['task_name', 'mean', 'std'])
        ew.writerow(['task_name', 'mean', 'std'])
        for task in tasks:
            fp = output_root / f'{task}_timing.csv'
            if not fp.exists():
                continue
            with open(fp, 'r', newline='') as rf:
                rows = list(csv.DictReader(rf))
            eval_vals = [float(r['evaluation_time']) for r in rows]
            train_vals = [float(r['training_cycle_time']) for r in rows]
            tw.writerow([task, statistics.mean(train_vals), statistics.pstdev(train_vals)])
            ew.writerow([task, statistics.mean(eval_vals), statistics.pstdev(eval_vals)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().with_name('timing.yaml')))
    parser.add_argument('--domain', required=True, choices=list(DOMAIN_TASKS.keys()))
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--obs', choices=['state', 'rgb'], default=None)
    parser.add_argument('--planner_type', choices=['mppi', 'diffusion'], default=None)
    parser.add_argument('--workers_per_gpu', type=int, default=None)
    parser.add_argument('--compile', choices=['true', 'false'], default=None)
    parser.add_argument('--compile_mode', default=None)
    parser.add_argument('--diffusion_eval_compile', choices=['true', 'false'], default=None)
    parser.add_argument('--diffusion_eval_compile_mode', default=None)
    parser.add_argument('--save_dir', default='/media/datasets/cheliu21/cxy_worldmodel/profiling/')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
        cfg.gpu_ids = []
    if args.planner_type is not None:
        cfg.planner_type = args.planner_type
    if args.workers_per_gpu is not None:
        cfg.workers_per_gpu = args.workers_per_gpu
    if args.obs is not None:
        cfg.obs = args.obs
    if args.compile is not None:
        cfg.compile = args.compile.lower() == 'true'
    if args.compile_mode is not None:
        cfg.compile_mode = args.compile_mode
    if args.diffusion_eval_compile is not None:
        cfg.diffusion_eval_compile = args.diffusion_eval_compile.lower() == 'true'
    if args.diffusion_eval_compile_mode is not None:
        cfg.diffusion_eval_compile_mode = args.diffusion_eval_compile_mode
    tasks = list(DOMAIN_TASKS[args.domain])
    gpu_ids = _resolve_gpu_ids(cfg)
    workers_per_gpu = int(cfg.get('workers_per_gpu', 3) or 3)
    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    q = _build_task_queue(tasks)
    failures: list[tuple[int, str, int]] = []
    threads = []
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            t = threading.Thread(target=_run_worker, args=(cfg, gpu_id, q, output_root, failures), daemon=False)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    if failures:
        raise SystemExit(f'failed tasks: {failures}')

    _aggregate_and_save(args.domain, tasks, output_root)
    print(f'[done] profiling csv saved under: {output_root}')


if __name__ == '__main__':
    main()
