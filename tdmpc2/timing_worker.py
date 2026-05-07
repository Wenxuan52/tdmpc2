import csv
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from common.buffer import Buffer
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')


def to_td(env, obs, action=None, reward=None, terminated=None):
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device='cpu')
    else:
        obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full_like(env.rand_act(), float('nan'))
    if reward is None:
        reward = torch.tensor(float('nan'))
    if terminated is None:
        terminated = torch.tensor(float('nan'))
    return TensorDict(obs=obs, action=action.unsqueeze(0), reward=reward.unsqueeze(0), terminated=terminated.unsqueeze(0), batch_size=(1,))


def eval_once(env, agent, eval_episodes):
    for _ in range(eval_episodes):
        obs, done, t = env.reset(), False, 0
        while not done:
            action = agent.act(obs, t0=t == 0, eval_mode=True)
            obs, _, done, _ = env.step(action)
            t += 1


@hydra.main(config_name='config', config_path='.')
def main(cfg):
    assert torch.cuda.is_available()
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    cfg = parse_cfg(cfg)
    set_seed(int(getattr(cfg, 'seed', 1)))

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)

    warmup_evals = int(getattr(cfg, 'warmup_evals', 2) or 2)
    measure_evals = int(getattr(cfg, 'measure_evals', 10) or 10)
    out_dir = Path(str(getattr(cfg, 'timing_output_dir', '.')))
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    obs, done = env.reset(), False
    tds = [to_td(env, obs)]

    for _ in range(warmup_evals):
        while step <= cfg.seed_steps + 10:
            if done:
                episode = torch.cat(tds)
                buffer.add(episode)
                obs, done = env.reset(), False
                tds = [to_td(env, obs)]
            action = agent.act(obs, t0=len(tds) == 1) if step > cfg.seed_steps else env.rand_act()
            obs, reward, done, info = env.step(action)
            tds.append(to_td(env, obs, action, reward, info['terminated']))
            if step >= cfg.seed_steps:
                if step == cfg.seed_steps:
                    for _ in range(cfg.seed_steps):
                        agent.update(buffer)
                else:
                    agent.update(buffer)
            step += 1
        eval_once(env, agent, cfg.eval_episodes)

    eval_times, train_cycle_times = [], []
    for _ in range(measure_evals):
        train_start = time.perf_counter()
        while step <= cfg.seed_steps + 50:
            if done:
                episode = torch.cat(tds)
                buffer.add(episode)
                obs, done = env.reset(), False
                tds = [to_td(env, obs)]
            action = agent.act(obs, t0=len(tds) == 1) if step > cfg.seed_steps else env.rand_act()
            obs, reward, done, info = env.step(action)
            tds.append(to_td(env, obs, action, reward, info['terminated']))
            if step >= cfg.seed_steps:
                agent.update(buffer)
            step += 1
        train_cycle_times.append(time.perf_counter() - train_start)

        eval_start = time.perf_counter()
        eval_once(env, agent, cfg.eval_episodes)
        eval_times.append(time.perf_counter() - eval_start)

    out_fp = out_dir / f'{cfg.task}_timing.csv'
    with open(out_fp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'training_cycle_time', 'evaluation_time'])
        for i, (tr, ev) in enumerate(zip(train_cycle_times, eval_times), 1):
            w.writerow([i, tr, ev])

    print(f'{cfg.task}: train mean={np.mean(train_cycle_times):.4f}s std={np.std(train_cycle_times):.4f}s | eval mean={np.mean(eval_times):.4f}s std={np.std(eval_times):.4f}s')


if __name__ == '__main__':
    main()
