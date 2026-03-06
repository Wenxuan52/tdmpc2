import math

import torch
import torch.nn as nn


class DiffusionPolicy(nn.Module):
	"""Simple epsilon-network over latent + noisy action trajectory + timestep."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.horizon = int(cfg.horizon)
		self.action_dim = int(cfg.action_dim)
		self.input_action_dim = self.horizon * self.action_dim
		self.time_dim = 64
		task_dim = int(cfg.task_dim) if cfg.multitask else 0
		self._task_emb = nn.Embedding(len(cfg.tasks), task_dim) if cfg.multitask else None

		in_dim = cfg.latent_dim + self.input_action_dim + self.time_dim + task_dim
		hid = min(1024, max(256, cfg.mlp_dim // 2))
		self.epsilon_net = nn.Sequential(
			nn.Linear(in_dim, hid),
			nn.SiLU(),
			nn.Linear(hid, hid),
			nn.SiLU(),
			nn.Linear(hid, self.input_action_dim),
		)

		betas = torch.linspace(cfg.diffusion_beta0, cfg.diffusion_betaT, int(cfg.diffusion_steps), dtype=torch.float32)
		alphas = 1.0 - betas
		alpha_bar = torch.cumprod(alphas, dim=0)
		self.register_buffer('betas', betas)
		self.register_buffer('alphas', alphas)
		self.register_buffer('alpha_bar', alpha_bar)

	def _time_embed(self, tau, dim=64):
		tau = tau.float().unsqueeze(-1)
		half = dim // 2
		freqs = torch.exp(
			torch.arange(half, device=tau.device, dtype=tau.dtype) * (-math.log(10000.0) / max(half - 1, 1))
		)
		angles = tau * freqs
		emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
		if dim % 2 == 1:
			emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
		return emb

	def q_sample(self, a0, tau, eps):
		tau = tau.long().clamp(1, self.alpha_bar.shape[0] - 1)
		alpha_bar_tau = self.alpha_bar[tau].view(-1, 1, 1)
		return torch.sqrt(alpha_bar_tau) * a0 + torch.sqrt(1.0 - alpha_bar_tau) * eps

	def forward(self, z, a_tau, tau, task=None):
		B = z.shape[0]
		a_flat = a_tau.reshape(B, -1)
		tau_emb = self._time_embed(tau, self.time_dim)
		x = [z, a_flat, tau_emb]
		if self._task_emb is not None:
			x.append(self._task_emb(task.long()))
		x = torch.cat(x, dim=-1)
		eps_hat = self.epsilon_net(x).view(B, self.horizon, self.action_dim)
		return torch.nan_to_num(eps_hat)

	@torch.no_grad()
	def sample(self, z, task=None, action_mask=None, steps=None, deterministic=True):
		_ = deterministic
		T = int(self.alpha_bar.shape[0])
		if steps is None:
			steps = T - 1
		steps = max(1, min(int(steps), T - 1))
		taus = torch.linspace(T - 1, 1, steps, device=z.device).long().unique_consecutive()
		x = torch.randn(z.shape[0], self.horizon, self.action_dim, device=z.device)

		for i, tau in enumerate(taus):
			tau_batch = torch.full((z.shape[0],), int(tau.item()), device=z.device, dtype=torch.long)
			eps_hat = self.forward(z, x, tau_batch, task=task)
			alpha_bar_tau = self.alpha_bar[tau_batch].view(-1, 1, 1)
			x0_hat = (x - torch.sqrt(1.0 - alpha_bar_tau) * eps_hat) / torch.sqrt(alpha_bar_tau)
			x0_hat = x0_hat.clamp(-1, 1)
			if i == len(taus) - 1:
				x = x0_hat
				break
			tau_prev = taus[i + 1]
			alpha_bar_prev = self.alpha_bar[tau_prev].view(1, 1, 1)
			x = torch.sqrt(alpha_bar_prev) * x0_hat + torch.sqrt(1.0 - alpha_bar_prev) * eps_hat
			if action_mask is not None:
				x = x * action_mask.view(1, 1, -1)
			x = x.clamp(-1, 1)

		if action_mask is not None:
			x = x * action_mask.view(1, 1, -1)
		return x
