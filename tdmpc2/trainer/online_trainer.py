import csv
import random
from time import time
from pathlib import Path

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from common.replay_buffer_saver import ReplayBufferSaver
from trainer.base import Trainer

MANISKILL2_TASKS = {'lift-cube', 'pick-cube', 'stack-cube', 'pick-ycb', 'turn-faucet'}


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._replay_saver = ReplayBufferSaver(self.cfg)
		self._reward_csv_fp = None
		self._reward_csv_mode = None
		self._csv_eval_freq = int(getattr(self.cfg, 'csv_eval_freq', 50_000) or 50_000)
		if bool(getattr(self.cfg, 'save_reward_csv', False)):
			reward_csv_dir = Path(str(getattr(self.cfg, 'reward_csv_dir', '')))
			reward_csv_dir.mkdir(parents=True, exist_ok=True)
			self._reward_csv_fp = reward_csv_dir / f'{self.cfg.task}_{self.cfg.seed}.csv'
			save_eval_success = (
				str(self.cfg.task).startswith('mw-')
				or str(self.cfg.task).startswith('myo-')
				or str(self.cfg.task) in MANISKILL2_TASKS
			)
			self._reward_csv_mode = 'eval_success' if save_eval_success else 'eval_reward'
			with open(self._reward_csv_fp, 'w', newline='') as f:
				writer = csv.writer(f)
				if self._reward_csv_mode == 'eval_success':
					writer.writerow(['step', 'episode_success'])
				else:
					writer.writerow(['step', 'episode_reward'])
		self._enable_diff_metrics = bool(getattr(self.cfg, 'enable_diff_metrics', False))
		self._drift_log_interval = int(getattr(self.cfg, 'drift_log_interval', 1000) or 1000)
		self._num_eval_states = int(getattr(self.cfg, 'num_eval_states', 1024) or 1024)
		self._min_buffer_size_for_eval = int(getattr(self.cfg, 'min_buffer_size_for_eval', 5000) or 5000)
		self._planner_for_drift = str(getattr(self.cfg, 'planner_type', 'mppi')).lower()
		self._eval_seed = int(getattr(self.cfg, 'drift_eval_seed', 0) or 0)
		self._eval_states = None
		self._prev_actions = None
		self._diff_metric_csv_fp = None
		self._diff_metric_fieldnames = [
			'step',
			'planner_gap/mppi_to_policy',
			'planner_gap/diffusion_to_policy',
			'action_drift/pi',
			'action_drift/mppi',
			'action_drift/diffusion',
		]
		self._enable_td_error_metrics = bool(getattr(self.cfg, 'enable_td_error_metrics', False))
		self._td_error_log_interval = int(getattr(self.cfg, 'td_error_log_interval', 1000) or 1000)
		self._td_metric_csv_fp = None
		self._td_metric_fieldnames = [
			'step',
			'mppi/mppi_td_error',
			'mppi/mppi_cross_td_error',
			'diffusion/diffusion_td_error',
			'diffusion/diffusion_cross_td_error',
		]
		if self._enable_diff_metrics:
			base_metric_root = str(getattr(self.cfg, 'diff_metric_root', '') or '').strip()
			if not base_metric_root:
				base_metric_root = str(getattr(self.cfg, 'replay_save_dir', '') or '.')
			metric_root = Path(base_metric_root)
			metric_root.mkdir(parents=True, exist_ok=True)
			task_name = str(self.cfg.task).replace('/', '_')
			self._diff_metric_csv_fp = metric_root / f'DIFF_metric_{task_name}_seed{self.cfg.seed}.csv'
			with open(self._diff_metric_csv_fp, 'w', newline='') as f:
				writer = csv.DictWriter(f, fieldnames=self._diff_metric_fieldnames)
				writer.writeheader()


		if self._enable_td_error_metrics:
			td_metric_root = Path('/media/datasets/cheliu21/cxy_worldmodel/tderror_metric')
			td_metric_root.mkdir(parents=True, exist_ok=True)
			task_name = str(self.cfg.task).replace('/', '_')
			self._td_metric_csv_fp = td_metric_root / f'TD_error_metric_{task_name}_seed{self.cfg.seed}.csv'
			with open(self._td_metric_csv_fp, 'w', newline='') as f:
				writer = csv.DictWriter(f, fieldnames=self._td_metric_fieldnames)
				writer.writeheader()
	def _append_reward_csv(self, step, reward):
		if self._reward_csv_fp is None:
			return
		with open(self._reward_csv_fp, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([int(step), float(reward)])

	def _append_eval_success_csv(self, step, success):
		if self._reward_csv_fp is None:
			return
		with open(self._reward_csv_fp, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([int(step), float(success)])

	def _try_build_eval_states(self):
		if self._eval_states is not None:
			return
		if self._step < self._min_buffer_size_for_eval:
			return
		chunks = []
		remaining = self._num_eval_states
		while remaining > 0:
			obs, *_ = self.buffer.sample()
			cur = obs[0].detach().cpu()
			if cur.ndim == 1:
				cur = cur.unsqueeze(0)
			take = min(remaining, cur.shape[0])
			chunks.append(cur[:take])
			remaining -= take
		self._eval_states = torch.cat(chunks, dim=0)

	@staticmethod
	def _mean_squared_action_diff(a, b):
		return ((a - b) ** 2).mean().item()

	@torch.no_grad()
	def _collect_actions_for_drift(self, states):
		was_training = self.agent.model.training
		self.agent.model.eval()
		torch.manual_seed(self._eval_seed)
		np.random.seed(self._eval_seed)
		random.seed(self._eval_seed)
		actions = {
			'pi': [],
			'mppi': [],
			'diff': [],
		}
		for state in states:
			actions['pi'].append(self.agent.act_policy(state, eval_mode=True))
			if self._planner_for_drift in {'mppi', 'both'}:
				actions['mppi'].append(self.agent.act_mppi(state, t0=True, eval_mode=True))
			if self._planner_for_drift in {'diffusion', 'both'}:
				actions['diff'].append(self.agent.act_diffusion(state, t0=True, eval_mode=True))
		if was_training:
			self.agent.model.train()
		for key, vals in actions.items():
			if vals:
				actions[key] = torch.stack(vals, dim=0).clamp(-1.0, 1.0)
			else:
				actions[key] = None
		return actions

	def _compute_action_drift_metrics(self, current_actions):
		metrics = {}
		if current_actions.get('mppi') is not None:
			metrics['planner_gap/mppi_to_policy'] = self._mean_squared_action_diff(
				current_actions['mppi'], current_actions['pi'])
		if current_actions.get('diff') is not None:
			metrics['planner_gap/diffusion_to_policy'] = self._mean_squared_action_diff(
				current_actions['diff'], current_actions['pi'])
		if self._prev_actions is not None:
			metrics['action_drift/pi'] = self._mean_squared_action_diff(current_actions['pi'], self._prev_actions['pi'])
			if current_actions.get('mppi') is not None and self._prev_actions.get('mppi') is not None:
				metrics['action_drift/mppi'] = self._mean_squared_action_diff(
					current_actions['mppi'], self._prev_actions['mppi'])
			if current_actions.get('diff') is not None and self._prev_actions.get('diff') is not None:
				metrics['action_drift/diffusion'] = self._mean_squared_action_diff(
					current_actions['diff'], self._prev_actions['diff'])
		return metrics

	def _append_diff_metric_csv(self, step, metrics):
		if self._diff_metric_csv_fp is None:
			return
		row = {'step': int(step)}
		for key in self._diff_metric_fieldnames[1:]:
			row[key] = float(metrics[key]) if key in metrics else float('nan')
		with open(self._diff_metric_csv_fp, 'a', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=self._diff_metric_fieldnames)
			writer.writerow(row)

	def _maybe_log_diff_metrics(self, train_metrics):
		if not self._enable_diff_metrics:
			return
		if self._drift_log_interval <= 0 or self._step == 0 or self._step % self._drift_log_interval != 0:
			return
		self._try_build_eval_states()
		if self._eval_states is None:
			return
		current_actions = self._collect_actions_for_drift(self._eval_states)
		metrics = self._compute_action_drift_metrics(current_actions)
		self._prev_actions = {
			k: (v.clone() if v is not None else None)
			for k, v in current_actions.items()
		}
		if not metrics:
			return
		self._append_diff_metric_csv(self._step, metrics)
		train_metrics.update(metrics)
		log_payload = dict(metrics)
		log_payload.update(self.common_metrics())
		self.logger.log(log_payload, 'train')


	def _append_td_metric_csv(self, step, metrics):
		if self._td_metric_csv_fp is None:
			return
		row = {'step': int(step)}
		for key in self._td_metric_fieldnames[1:]:
			row[key] = float(metrics[key]) if key in metrics else float('nan')
		with open(self._td_metric_csv_fp, 'a', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=self._td_metric_fieldnames)
			writer.writerow(row)

	@torch.no_grad()
	def _compute_td_error_metrics(self):
		obs, action, reward, terminated, task = self.buffer.sample()
		obs_t = obs[0]
		next_obs = obs[1]
		a_t = action[0]
		r_t = reward[0]
		d_t = terminated[0]
		task_t = task if task is None else task[0]

		z_t = self.agent.model.encode(obs_t, task_t)
		next_z = self.agent.model.encode(next_obs, task_t)
		q_pred = self.agent.model.Q(z_t, a_t, task_t, return_type='avg')
		if q_pred.ndim == 3:
			q_pred = q_pred.mean(dim=0)

		discount = self.agent.discount[task_t].unsqueeze(-1) if self.cfg.multitask else self.agent.discount
		planner = str(getattr(self.cfg, 'planner_type', 'mppi')).lower()

		if planner == 'diffusion':
			beta_action = action[1]
			search_actions = [self.agent.act_diffusion(state, t0=True, eval_mode=True) for state in next_obs.detach().cpu()]
			search_action = torch.stack(search_actions, dim=0).to(self.agent.device, non_blocking=True)
		else:
			beta_action, _ = self.agent.model.pi(next_z, task_t)
			search_actions = [self.agent.act_mppi(state, t0=True, eval_mode=True) for state in next_obs.detach().cpu()]
			search_action = torch.stack(search_actions, dim=0).to(self.agent.device, non_blocking=True)

		q_beta_all = self.agent.model.Q(next_z, beta_action, task_t, return_type='all', target=True)
		q_beta = torch.min(q_beta_all[0], q_beta_all[1])
		y_beta = r_t + discount * (1 - d_t) * q_beta
		delta_beta = y_beta - q_pred

		q_search_all = self.agent.model.Q(next_z, search_action, task_t, return_type='all', target=True)
		q_search = torch.min(q_search_all[0], q_search_all[1])
		y_search = r_t + discount * (1 - d_t) * q_search

		td_error = delta_beta.abs().mean().item()
		cross_gap = (y_search - y_beta).abs().mean().item()
		metrics = {}
		if planner == 'diffusion':
			metrics['diffusion/diffusion_td_error'] = td_error
			metrics['diffusion/diffusion_cross_td_error'] = cross_gap
		else:
			metrics['mppi/mppi_td_error'] = td_error
			metrics['mppi/mppi_cross_td_error'] = cross_gap
		return metrics


	def _maybe_log_td_error_metrics(self, train_metrics):
		if not self._enable_td_error_metrics:
			return
		if self._td_error_log_interval <= 0 or self._step == 0 or self._step % self._td_error_log_interval != 0:
			return
		if self._step < self._min_buffer_size_for_eval:
			return
		metrics = self._compute_td_error_metrics()
		if not metrics:
			return
		self._append_td_metric_csv(self._step, metrics)
		train_metrics.update(metrics)
		log_payload = dict(metrics)
		log_payload.update(self.common_metrics())
		self.logger.log(log_payload, 'train')

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		eval_freq = int(getattr(self.cfg, 'eval_freq', 0) or 0)
		save_model_every = int(getattr(self.cfg, 'save_model_every', 100_000) or 0)
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if eval_freq > 0 and self._step % eval_freq == 0:
				eval_next = True
				if self._reward_csv_mode in {'eval_success', 'eval_reward'} and self._step > 0 and self._step % self._csv_eval_freq == 0:
					eval_next = True

			# Save model periodically (independent from eval)
			if save_model_every > 0 and self._step > 0 and self._step % save_model_every == 0:
				self.logger.save_agent(self.agent, identifier=f'{self._step}')

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					if self._reward_csv_mode == 'eval_success':
						self._append_eval_success_csv(self._step, eval_metrics['episode_success'])
					elif self._reward_csv_mode == 'eval_reward':
						self._append_reward_csv(self._step, eval_metrics['episode_reward'])
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					episode_td = torch.cat(self._tds)
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._replay_saver.add_episode(episode_td)
					self._ep_idx = self.buffer.add(episode_td)

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)
				self._maybe_log_diff_metrics(train_metrics)
				self._maybe_log_td_error_metrics(train_metrics)

			self._step += 1

		self._replay_saver.finish()
		self.logger.finish(self.agent)
