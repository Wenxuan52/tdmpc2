"""Smoke test for TDMPC2 model DDP wrapping.

Run from repo root:
  python tdmpc2/tests/ddp_model_wrap_smoke.py
"""

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tdmpc2.tdmpc2 import TDMPC2


class _DummyWorldModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		dim = 4
		self._encoder = nn.Linear(dim, dim)
		self._dynamics = nn.Linear(dim, dim)
		self._reward = nn.Linear(dim, 1)
		self._termination = nn.Linear(dim, 1)
		self._F = nn.Linear(dim, 1)
		self._Qs = nn.ModuleList([nn.Linear(dim, 1)])
		self._task_emb = nn.Embedding(1, 1)
		self._pi = nn.Linear(dim, cfg.action_dim)
		self.register_buffer("_action_masks", torch.ones(1, cfg.action_dim))

	def eval(self):
		return super().eval()

	def encode(self, obs, task=None):
		return self._encoder(obs)


def _build_cfg():
	return SimpleNamespace(
		device='cpu',
		use_ddp=True,
		local_rank=0,
			ddp_find_unused_parameters=False,
			ddp_use_native_wrapper=True,
			ddp_broadcast_buffers=False,
		ddp_gradient_as_bucket_view=True,
		ddp_static_graph=False,
		lr=1e-3,
		enc_lr_scale=1.0,
		episodic=False,
		multitask=False,
		iterations=1,
		action_dim=2,
		horizon=1,
		compile=False,
		compile_mode='reduce-overhead',
		contrastive_beta=0.2,
		contrastive_coef=1.0,
		contrastive_clip=5.0,
		contrastive_momentum=0.99,
		episode_length=10,
		episode_lengths=[10],
		discount_denom=5,
		discount_min=0.95,
		discount_max=0.995,
	)


def test_ddp_model_wrap():
	with tempfile.TemporaryDirectory() as tmp_dir:
		rendezvous_path = Path(tmp_dir) / 'ddp_model_wrap_rendezvous'
		rendezvous_path.touch(exist_ok=True)
		dist.init_process_group(
			backend='gloo',
			init_method=f'file://{rendezvous_path}',
			rank=0,
			world_size=1,
		)
		try:
			with patch('tdmpc2.tdmpc2.WorldModel', _DummyWorldModel), \
				 patch('tdmpc2.tdmpc2.DiffusionPlanner', lambda cfg: object()):
				cfg = _build_cfg()
				agent = TDMPC2(cfg)
				assert isinstance(agent.model, DDP)
				assert isinstance(agent._raw_model, _DummyWorldModel)
				assert agent.pi_optim.param_groups[0]['params'][0] is agent._raw_model._pi.weight
				obs = torch.zeros(1, 4)
				z = agent._model_call("encode", obs, None)
				assert z.shape == (1, 4)
		finally:
			dist.destroy_process_group()
			rendezvous_path.unlink(missing_ok=True)


if __name__ == '__main__':
	test_ddp_model_wrap()
	print('ddp_model_wrap_smoke: all checks passed')
