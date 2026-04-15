import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import torch
from termcolor import colored

from common import TASK_SET
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from off2on import (
    SOURCE_TASK_MAP,
    _append_new_task_embedding,
    _infer_mt80_dims,
    _load_checkpoint_state_dict,
)

torch.backends.cudnn.benchmark = True

OFF2ON_TASKS = [
    'walker-run',
    'cheetah-run',
    'hopper-hop',
    'pendulum-swingup',
    'reacher-hard',
    'mw-bin-picking',
    'mw-box-close',
    'mw-door-lock',
    'mw-door-unlock',
    'mw-hand-insert',
]


def _episode_length(task_name: str) -> int:
    return 100 if task_name.startswith('mw-') else 500


def _normalized_score(task: str, ep_reward: float, ep_success: float) -> float:
    return ep_success * 100.0 if task.startswith('mw-') else ep_reward / 10.0


def _collect_models(task_dir: Path):
    return sorted(task_dir.glob('final_*.pt'))


def _build_agent_for_task(cfg, model_path: Path, target_task: str):
    cfg = deepcopy(cfg)
    cfg.task = 'mt80'
    cfg.multitask = True
    cfg.task_dim = 96
    cfg.tasks = list(TASK_SET['mt80'])

    state_dict = _load_checkpoint_state_dict(str(model_path))
    obs_dim, action_dim, action_dims = _infer_mt80_dims(state_dict, cfg.task_dim)
    # `state_dict` is an off2on model with an extra task-id row; keep base mt80 dims here.
    if len(action_dims) >= len(cfg.tasks):
        action_dims = action_dims[:len(cfg.tasks)]
    else:
        action_dims = [action_dim for _ in cfg.tasks]

    cfg.obs = 'state'
    cfg.obs_shape = {'state': (obs_dim,)}
    cfg.action_dim = action_dim
    cfg.action_dims = action_dims
    cfg.episode_lengths = [_episode_length(t) for t in cfg.tasks]
    cfg.episode_length = _episode_length(target_task)

    agent = TDMPC2(cfg)
    new_task_idx = _append_new_task_embedding(
        agent,
        target_task=target_task,
        source_task=SOURCE_TASK_MAP[target_task],
    )
    agent.load(str(model_path))
    return agent, new_task_idx


@hydra.main(config_name='config', config_path='.')
def evaluate_off2on(cfg):
    """Evaluate off2on checkpoints under a root dir and report per-task + average score."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    root = Path(str(cfg.get('off2on_eval_root', '') or '').strip())
    if not root:
        raise ValueError('Please provide `off2on_eval_root=/path/to/off2on_outputs`')
    if not root.exists():
        raise FileNotFoundError(f'Root not found: {root}')

    eval_episodes = int(getattr(cfg, 'eval_episodes', 10) or 10)
    if eval_episodes <= 0:
        raise ValueError('`eval_episodes` must be > 0')

    print(colored(f'Off2On root: {root}', 'blue', attrs=['bold']))
    print(colored(f'Eval episodes per model: {eval_episodes}', 'blue', attrs=['bold']))

    task_scores = []
    for task in OFF2ON_TASKS:
        task_dir = root / task
        model_fps = _collect_models(task_dir)
        if len(model_fps) == 0:
            print(colored(f'[SKIP] {task:<16} no final_*.pt found under {task_dir}', 'yellow'))
            continue

        _cfg = deepcopy(cfg)
        _cfg.multitask = False
        _cfg.task = task
        env = make_env(_cfg)

        seed_scores = []
        for model_fp in model_fps:
            agent, task_idx = _build_agent_for_task(cfg, model_fp, task)
            ep_rewards, ep_successes = [], []
            for _ in range(eval_episodes):
                obs, done, ep_reward, t = env.reset(), False, 0.0, 0
                while not done:
                    action = agent.act(obs, t0=(t == 0), eval_mode=True, task=task_idx)
                    obs, reward, done, info = env.step(action[: env.action_space.shape[0]])
                    ep_reward += float(reward)
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(float(info['success']))

            mean_reward = float(np.mean(ep_rewards))
            mean_success = float(np.mean(ep_successes))
            score = _normalized_score(task, mean_reward, mean_success)
            seed_scores.append(score)

            print(colored(
                f'  {task:<16} | {model_fp.name:<12} | R={mean_reward:8.2f} S={mean_success:6.3f} Score={score:7.2f}',
                'yellow'))

        if seed_scores:
            task_mean = float(np.mean(seed_scores))
            task_scores.append(task_mean)
            print(colored(f'[TASK] {task:<16} mean score over models: {task_mean:.2f}', 'green', attrs=['bold']))

    if not task_scores:
        raise RuntimeError(f'No valid off2on models found under: {root}')

    avg_score = float(np.mean(task_scores))
    print(colored(f'Average score over {len(task_scores)} tasks: {avg_score:.2f}', 'green', attrs=['bold']))


if __name__ == '__main__':
    evaluate_off2on()