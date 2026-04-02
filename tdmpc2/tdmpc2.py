import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tdmpc2.common import math
from tdmpc2.common.scale import RunningScale
from tdmpc2.common.world_model import WorldModel
from tdmpc2.common.layers import api_model_conversion
from tdmpc2.planners import DiffusionPlanner
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		default_device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}' if torch.cuda.is_available() else 'cpu'
		self.device = torch.device(str(getattr(cfg, 'device', default_device)))
		base_model = WorldModel(cfg).to(self.device)
		def _dispatch_forward(call_name, *args, **kwargs):
			return getattr(base_model, call_name)(*args, **kwargs)
		base_model.forward = _dispatch_forward
		self.model = base_model
		self._ddp_enabled = bool(getattr(cfg, 'use_ddp', False)) and dist.is_available() and dist.is_initialized()
		self._use_native_ddp_wrapper = self._ddp_enabled and bool(getattr(cfg, 'ddp_use_native_wrapper', False))
		if self._use_native_ddp_wrapper:
			local_rank = int(getattr(cfg, 'local_rank', os.environ.get('LOCAL_RANK', 0)))
			ddp_kwargs = {
				'find_unused_parameters': bool(getattr(cfg, 'ddp_find_unused_parameters', True)),
				'broadcast_buffers': bool(getattr(cfg, 'ddp_broadcast_buffers', False)),
				'gradient_as_bucket_view': bool(getattr(cfg, 'ddp_gradient_as_bucket_view', True)),
			}
			if self.device.type == 'cuda':
				ddp_kwargs['device_ids'] = [local_rank]
				ddp_kwargs['output_device'] = local_rank
			self.model = DDP(base_model, **ddp_kwargs)
			if bool(getattr(cfg, 'ddp_static_graph', False)):
				self.model._set_static_graph()

		model_for_optim = self._raw_model
		self.optim = torch.optim.Adam([
			{'params': model_for_optim._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': model_for_optim._dynamics.parameters()},
			{'params': model_for_optim._reward.parameters()},
			{'params': model_for_optim._termination.parameters() if self.cfg.episodic else []},
			{'params': model_for_optim._F.parameters()},
			{'params': model_for_optim._Qs.parameters()},
			{'params': model_for_optim._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(model_for_optim._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self._compile_mode = str(getattr(cfg, 'compile_mode', 'max-autotune-no-cudagraphs'))
		self._contrastive_beta = float(getattr(cfg, 'contrastive_beta', 0.2))
		self._contrastive_coef = float(getattr(cfg, 'contrastive_coef', 1.0))
		self._contrastive_clip = float(getattr(cfg, 'contrastive_clip', 5.0))
		self._contrastive_momentum = float(getattr(cfg, 'contrastive_momentum', 0.99))
		self.register_buffer("_contrastive_mean", torch.zeros(1, device=self.device))
		self.register_buffer("_contrastive_std", torch.ones(1, device=self.device))
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self._diffusion_planner = DiffusionPlanner(cfg)
		self._compiled_update_core = self._update_core
		if cfg.compile:
			print(f'Compiling update core with torch.compile (mode={self._compile_mode})...')
			self._compiled_update_core = torch.compile(self._update_core, mode=self._compile_mode)

	@property
	def _raw_model(self):
		return self.model.module if isinstance(self.model, DDP) else self.model

	def _model_call(self, call_name, *args, **kwargs):
		if isinstance(self.model, DDP):
			return self.model(call_name, *args, **kwargs)
		return getattr(self.model, call_name)(*args, **kwargs)

	def _sync_gradients(self, params):
		if not self._ddp_enabled:
			return
		world_size = dist.get_world_size()
		if world_size <= 1:
			return
		for p in params:
			grad = p.grad if p.grad is not None else torch.zeros_like(p)
			dist.all_reduce(grad, op=dist.ReduceOp.SUM)
			if p.grad is not None:
				p.grad.copy_(grad.div(world_size))

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		planner_type = self.cfg.get('planner_type', 'mppi')
		if planner_type == 'diffusion':
			plan = self._diffusion_plan
		elif self.cfg.compile:
			plan = torch.compile(self._plan, mode=self._compile_mode)
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _diffusion_plan(self, obs, t0=False, eval_mode=False, task=None):
		return self._diffusion_planner.plan(self, obs, t0=t0, eval_mode=eval_mode, task=task)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self._raw_model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self._raw_model.state_dict(), state_dict)
		missing_keys, unexpected_keys = self._raw_model.load_state_dict(state_dict, strict=False)

		# Backward compatibility: `_G` was removed from the world model.
		allowed_missing_g_keys = tuple(key for key in missing_keys if key.startswith('_G.'))
		missing_keys = [key for key in missing_keys if key not in allowed_missing_g_keys]
		allowed_unexpected_g_keys = tuple(key for key in unexpected_keys if key.startswith('_G.'))
		unexpected_keys = [key for key in unexpected_keys if key not in allowed_unexpected_g_keys]
		allowed_f_keys = tuple(key for key in missing_keys if key.startswith('_F.'))
		missing_keys = [key for key in missing_keys if key not in allowed_f_keys]

		if missing_keys or unexpected_keys:
			pieces = []
			if missing_keys:
				pieces.append(f'Missing key(s) in state_dict: {missing_keys}')
			if unexpected_keys:
				pieces.append(f'Unexpected key(s) in state_dict: {unexpected_keys}')
			extra = ''
			if allowed_missing_g_keys or allowed_unexpected_g_keys:
				extra = (
					' Legacy checkpoint note: `_G` weights were ignored because `_G` is no longer part of WorldModel.'
				)
			raise RuntimeError('Error(s) in loading state_dict for WorldModel: ' + '; '.join(pieces) + extra)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self._model_call("encode", obs, task)
		action, info = self._model_call("pi", z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		num_samples = actions.shape[1]
		termination = torch.zeros(num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self._model_call("reward", z, actions[t], task), self.cfg)
			f_score = self._model_call("F", z, actions[t], task)
			f_norm = self._normalize_contrastive_score(f_score)
			shaped_reward = reward + self._contrastive_beta * f_norm
			z = self._model_call("next", z, actions[t], task)
			G = G + discount * (1-termination) * shaped_reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self._model_call("termination", z, task) > 0.5).float(), max=1.)
		action, _ = self._model_call("pi", z, task)
		return G + discount * (1-termination) * self._model_call("Q", z, action, task, return_type='avg')

	def _normalize_contrastive_score(self, score):
		std = torch.clamp(self._contrastive_std, min=1e-6)
		normed = (score - self._contrastive_mean) / std
		return normed.clamp(-self._contrastive_clip, self._contrastive_clip)

	@torch.no_grad()
	def _update_contrastive_stats(self, scores):
		batch_mean = scores.mean()
		batch_std = scores.std(unbiased=False)
		batch_std = torch.clamp(batch_std, min=1e-6)
		self._contrastive_mean.mul_(self._contrastive_momentum).add_(batch_mean * (1 - self._contrastive_momentum))
		self._contrastive_std.mul_(self._contrastive_momentum).add_(batch_std * (1 - self._contrastive_momentum))

	def _contrastive_loss(self, zs, actions, task):
		"""
		Binary contrastive objective: buffer actions as positives and model-generated actions as hard negatives.
		"""
		z_flat = zs.reshape(-1, zs.shape[-1]).detach()
		a_pos = actions.permute(1, 0, 2).reshape(-1, actions.shape[-1])
		task_flat = None
		if task is not None:
			task_flat = task.long().reshape(-1).repeat_interleave(self.cfg.horizon)

		pos_logits = self._model_call("F", z_flat, a_pos, task_flat)
		labels_pos = torch.ones_like(pos_logits)

		with torch.no_grad():
			pi_action, _ = self._model_call("pi", z_flat, task_flat)
			noise_scale = float(getattr(self.cfg, 'contrastive_neg_noise', 0.5))
			noisy_action = (pi_action + noise_scale * torch.randn_like(pi_action)).clamp(-1, 1)
			random_action = (2.0 * torch.rand_like(pi_action) - 1.0).clamp(-1, 1)
			candidate_actions = torch.stack([pi_action, noisy_action, random_action], dim=1)
			candidate_logits = self._model_call("F", 
				z_flat.unsqueeze(1).expand(-1, candidate_actions.shape[1], -1).reshape(-1, z_flat.shape[-1]),
				candidate_actions.reshape(-1, candidate_actions.shape[-1]),
				task_flat.repeat_interleave(candidate_actions.shape[1]) if task_flat is not None else None,
			).reshape(-1, candidate_actions.shape[1], 1).squeeze(-1)
			hard_idx = candidate_logits.argmax(dim=1)
			hard_neg = candidate_actions[torch.arange(candidate_actions.shape[0], device=hard_idx.device), hard_idx]

		neg_logits = self._model_call("F", z_flat, hard_neg, task_flat)
		labels_neg = torch.zeros_like(neg_logits)

		loss_pos = F.binary_cross_entropy_with_logits(pos_logits, labels_pos)
		loss_neg = F.binary_cross_entropy_with_logits(neg_logits, labels_neg)
		self._update_contrastive_stats(torch.cat([pos_logits.detach(), neg_logits.detach()], dim=0))
		return 0.5 * (loss_pos + loss_neg)

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self._model_call("encode", obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self._model_call("pi", _z, task)
				_z = self._model_call("next", _z, pi_actions[t], task)
			pi_actions[-1], _ = self._model_call("pi", _z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self._raw_model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self._raw_model._action_masks[task]
				std = std * self._raw_model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self._model_call("pi", zs, task)
		qs = self._model_call("Q", zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		if not self._use_native_ddp_wrapper:
			self._sync_gradients(self._raw_model._pi.parameters())
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self._raw_model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self._model_call("pi", next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		target_qs = self._model_call("Q", next_z, action, task, return_type='all', target=True)
		target_q = math.two_hot_inv(target_qs[:2], self.cfg).min(0).values
		return reward + discount * (1-terminated) * target_q

	def _update_core(self, obs, action, reward, terminated, task=None):
		with torch.no_grad():
			next_z = self._model_call("encode", obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self._model_call("encode", obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self._model_call("next", z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		_zs = zs[:-1]
		qs = self._model_call("Q", _zs, action, task, return_type='all')
		reward_preds = self._model_call("reward", _zs, action, task)
		termination_pred = self._model_call("termination", zs[1:], task, unnormalized=True) if self.cfg.episodic else None

		reward_loss, value_loss = 0, 0
		contrastive_loss = self._contrastive_loss(_zs, action, task)
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for qs_unbind_unbind in qs_unbind.unbind(0):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated) if self.cfg.episodic else reward_loss.new_zeros(())
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self._contrastive_coef * contrastive_loss +
			self.cfg.value_coef * value_loss
		)
		return zs.detach(), consistency_loss, reward_loss, value_loss, contrastive_loss, termination_loss, total_loss, termination_pred

	def _update(self, obs, action, reward, terminated, task=None):
		self.model.train()
		zs, consistency_loss, reward_loss, value_loss, contrastive_loss, termination_loss, total_loss, termination_pred = self._compiled_update_core(
			obs, action, reward, terminated, task,
		)

		# Update model
		total_loss.backward()
		if not self._use_native_ddp_wrapper:
			self._sync_gradients(self._raw_model.parameters())
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self._model_call("soft_update_target_Q")

		# Return training statistics
		self.model.eval()
		info_dict = {
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"contrastive_loss": contrastive_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		}
		info = TensorDict(info_dict)
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		return self._update(obs, action, reward, terminated, **kwargs)
