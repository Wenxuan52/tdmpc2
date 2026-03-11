import torch


class DiffusionPlanner:
	"""No-gradient model-based diffusion planner for TD-MPC2."""

	def __init__(self, cfg):
		self.cfg = cfg

	@torch.no_grad()
	def plan(self, agent, obs, t0=False, eval_mode=False, task=None):
		cfg = self.cfg
		device = agent.device

		num_steps = max(int(cfg.diffusion_steps), 2)
		num_samples = int(cfg.diffusion_num_samples)
		temperature = float(cfg.diffusion_temperature)
		action_noise = float(cfg.diffusion_action_noise)

		z0 = agent.model.encode(obs, task)
		z = z0.repeat(num_samples, 1)

		betas = torch.linspace(cfg.diffusion_beta0, cfg.diffusion_betaT, num_steps, device=device)
		alphas = 1.0 - betas
		alpha_bar = torch.cumprod(alphas, dim=0)

		if t0:
			mean0 = torch.zeros(cfg.horizon, cfg.action_dim, device=device)
		else:
			mean0 = torch.cat([
				agent._prev_mean[1:],
				torch.zeros_like(agent._prev_mean[:1]),
			], dim=0)

		x_tau = torch.sqrt(alpha_bar[-1]) * mean0
		action_mask = None
		if cfg.multitask:
			action_mask = agent.model._action_masks[task].squeeze(0)
			x_tau = x_tau * action_mask

		num_elites = int(getattr(cfg, 'diffusion_num_elites', 0) or 0)
		num_pi_trajs = int(getattr(cfg, 'diffusion_num_pi_trajs', 0) or 0)
		clamp_each_step = bool(getattr(cfg, 'diffusion_clamp_each_step', False))

		for tau in range(num_steps - 1, 0, -1):
			alpha_bar_tau = alpha_bar[tau]
			mean_cond = x_tau / torch.sqrt(alpha_bar_tau)
			std_cond = torch.sqrt((1.0 - alpha_bar_tau) / alpha_bar_tau)
			eps = torch.randn(num_samples, cfg.horizon, cfg.action_dim, device=device)
			a0_samples = mean_cond.unsqueeze(0) + std_cond * eps
			if action_mask is not None:
				a0_samples = a0_samples * action_mask
			a0_samples = a0_samples.clamp(-1, 1)

			if num_pi_trajs > 0:
				pi_trajs = min(num_pi_trajs, num_samples)
				pi_actions = torch.empty(pi_trajs, cfg.horizon, cfg.action_dim, device=device)
				z_pi = z0.repeat(pi_trajs, 1)
				for t in range(cfg.horizon):
					a_pi, _ = agent.model.pi(z_pi, task)
					if action_mask is not None:
						a_pi = a_pi * action_mask
					a_pi = a_pi.clamp(-1, 1)
					pi_actions[:, t] = a_pi
					z_pi = agent.model.next(z_pi, a_pi, task)
				a0_samples[:pi_trajs] = pi_actions

			actions_for_value = a0_samples.permute(1, 0, 2)
			values = agent._estimate_value(z, actions_for_value, task).nan_to_num(0.0).squeeze(-1)

			g_mean = values.mean()
			g_std = values.std()
			if g_std < 1e-4:
				g_std = torch.tensor(1.0, device=device)
			logits = (values - g_mean) / g_std
			logits = logits / max(temperature, 1e-6)
			if num_elites > 0 and num_elites < num_samples:
				elite_idx = torch.topk(values, num_elites, dim=0).indices
				elite_logits = logits[elite_idx]
				elite_logits = elite_logits - elite_logits.max()
				elite_weights = torch.softmax(elite_logits, dim=0)
				a_bar = (elite_weights[:, None, None] * a0_samples[elite_idx]).sum(dim=0)
			else:
				weights = torch.softmax(logits, dim=0)
				a_bar = (weights[:, None, None] * a0_samples).sum(dim=0)

			score_mb = (-x_tau + torch.sqrt(alpha_bar_tau) * a_bar) / (1.0 - alpha_bar_tau + 1e-8)
			x_tau = (x_tau + (1.0 - alpha_bar_tau) * score_mb) / torch.sqrt(alphas[tau])
			if action_mask is not None:
				x_tau = x_tau * action_mask
			if clamp_each_step:
				x_tau = x_tau.clamp(-1, 1)

		x0 = x_tau
		if action_mask is not None:
			x0 = x0 * action_mask
		x0 = x0.clamp(-1, 1)
		if cfg.get('diffusion_log_stats', False):
			print(
				f"[DiffusionPlanner] action_mean={x0.mean().item():.4f} action_std={x0.std().item():.4f} "
				f"score_mean={values.mean().item():.4f} score_std={values.std().item():.4f}"
			)
		agent._prev_mean.copy_(x0.detach())

		action = x0[0]
		if not eval_mode:
			action = action + torch.randn_like(action) * action_noise
			if action_mask is not None:
				action = action * action_mask
		action = action.clamp(-1, 1)
		return action
