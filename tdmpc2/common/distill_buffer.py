import torch


class DistillBuffer:
	"""Lightweight ring buffer for diffusion distillation trajectories."""

	def __init__(self, cfg):
		self.cfg = cfg
		self.capacity = int(cfg.diffusion_distill_buffer_size)
		self.device = torch.device('cpu')
		self.z = torch.zeros(self.capacity, cfg.latent_dim, dtype=torch.float32, device=self.device)
		self.a0 = torch.zeros(self.capacity, cfg.horizon, cfg.action_dim, dtype=torch.float32, device=self.device)
		self.task = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
		self.mask = torch.ones(self.capacity, cfg.action_dim, dtype=torch.float32, device=self.device)
		self.ptr = 0
		self.size = 0

	def __len__(self):
		return self.size

	def add(self, z0, a0_plan, task_idx=None, action_mask=None):
		idx = self.ptr
		self.z[idx].copy_(z0.detach().to(self.device, dtype=torch.float32))
		self.a0[idx].copy_(a0_plan.detach().to(self.device, dtype=torch.float32))
		if task_idx is None:
			self.task[idx] = 0
		else:
			self.task[idx] = int(task_idx)
		if action_mask is None:
			self.mask[idx].fill_(1.0)
		else:
			self.mask[idx].copy_(action_mask.detach().to(self.device, dtype=torch.float32))
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size, device):
		if self.size <= 0:
			raise ValueError('Cannot sample from an empty distillation buffer.')
		idxs = torch.randint(0, self.size, (int(batch_size),), device=self.device)
		return (
			self.z[idxs].to(device, non_blocking=True),
			self.a0[idxs].to(device, non_blocking=True),
			self.task[idxs].to(device, non_blocking=True),
			self.mask[idxs].to(device, non_blocking=True),
		)
