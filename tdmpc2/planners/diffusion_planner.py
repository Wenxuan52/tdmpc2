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
		mf_guidance = bool(cfg.get('diffusion_mf_guidance', False))
		mf_beta = float(cfg.get('diffusion_mf_beta', 0.0))
		mf_scale = float(cfg.get('diffusion_mf_scale', 1.0))
		mf_norm_clip = float(cfg.get('diffusion_mf_norm_clip', 0.0))
		mf_q_type = cfg.get('diffusion_mf_q_type', 'min')
		if mf_q_type not in {'min', 'avg'}:
			mf_q_type = 'min'
		mf_use_target_q = bool(cfg.get('diffusion_mf_use_target_q', False))
		use_mf_guidance = mf_guidance and mf_beta > 0.0

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

		q_value_log = []
		grad_norm_log = []
		for tau in range(num_steps - 1, 0, -1):
			alpha_bar_tau = alpha_bar[tau]
			mean_cond = x_tau / torch.sqrt(alpha_bar_tau)
			std_cond = torch.sqrt((1.0 - alpha_bar_tau) / alpha_bar_tau)
			eps = torch.randn(num_samples, cfg.horizon, cfg.action_dim, device=device)
			a0_samples = mean_cond.unsqueeze(0) + std_cond * eps
			if action_mask is not None:
				a0_samples = a0_samples * action_mask
			a0_samples = a0_samples.clamp(-1, 1)

			actions_for_value = a0_samples.permute(1, 0, 2)
			values = agent._estimate_value(z, actions_for_value, task).nan_to_num(0.0).squeeze(-1)

			g_mean = values.mean()
			g_std = values.std()
			if g_std < 1e-4:
				g_std = torch.tensor(1.0, device=device)
			logits = (values - g_mean) / g_std
			logits = logits / max(temperature, 1e-6)
			weights = torch.softmax(logits, dim=0)
			a_bar = (weights[:, None, None] * a0_samples).sum(dim=0)

			score_mb = (-x_tau + torch.sqrt(alpha_bar_tau) * a_bar) / (1.0 - alpha_bar_tau + 1e-8)
			score = score_mb
			if use_mf_guidance:
				with torch.enable_grad():
					a0 = a_bar[0].detach().clone().requires_grad_(True)
					if action_mask is not None:
						a0 = a0 * action_mask
					a0 = a0.clamp(-1, 1)
					q = agent.model.Q(
						z0.detach(),
						a0.unsqueeze(0),
						task,
						return_type=mf_q_type,
						target=mf_use_target_q,
						detach=False,
					)
					q_scalar = q.reshape(-1)
					grad_a0 = torch.autograd.grad(
						q_scalar.sum(),
						a0,
						retain_graph=False,
						create_graph=False,
						allow_unused=False,
					)[0].detach()

				if not torch.isfinite(grad_a0).all():
					grad_a0 = torch.zeros_like(grad_a0)
				grad_a0 = grad_a0 * mf_scale
				if mf_norm_clip > 0:
					grad_norm = grad_a0.norm(p=2) + 1e-8
					scale = min(1.0, mf_norm_clip / grad_norm.item())
					grad_a0 = grad_a0 * scale

				score_mf = torch.zeros_like(score_mb)
				score_mf[0] = grad_a0
				if action_mask is not None:
					score_mf = score_mf * action_mask
				score = (1.0 - mf_beta) * score_mb + mf_beta * score_mf

				if cfg.get('diffusion_log_stats', False):
					q_value_log.append(q_scalar.mean().detach())
					grad_norm_log.append(grad_a0.norm(p=2).detach())

			x_tau = (x_tau + (1.0 - alpha_bar_tau) * score) / torch.sqrt(alphas[tau])
			if action_mask is not None:
				x_tau = x_tau * action_mask
			x_tau = x_tau.clamp(-1, 1)

		x0 = x_tau
		if action_mask is not None:
			x0 = x0 * action_mask
		x0 = x0.clamp(-1, 1)
		if cfg.get('diffusion_log_stats', False):
			mf_debug = ""
			if len(q_value_log) > 0:
				q_vals = torch.stack(q_value_log)
				grad_norms = torch.stack(grad_norm_log)
				mf_debug = (
					f" mf_q_mean={q_vals.mean().item():.4f} mf_q_std={q_vals.std().item():.4f}"
					f" mf_grad_norm_mean={grad_norms.mean().item():.4f}"
				)
			print(
				f"[DiffusionPlanner] action_mean={x0.mean().item():.4f} action_std={x0.std().item():.4f} "
				f"score_mean={values.mean().item():.4f} score_std={values.std().item():.4f}{mf_debug}"
			)
		agent._prev_mean.copy_(x0.detach())

		action = x0[0]
		if not eval_mode:
			action = action + torch.randn_like(action) * action_noise
			if action_mask is not None:
				action = action * action_mask
		action = action.clamp(-1, 1)
		return action
