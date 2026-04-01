"""Smoke test for rank-aware logger behavior.

Run:
  python tdmpc2/tests/logger_rank0_smoke.py
"""

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tdmpc2.common.logger import Logger


def _cfg(work_dir, global_rank):
	cfg = SimpleNamespace(
		work_dir=Path(work_dir),
		global_rank=global_rank,
		rank0_logging_only=True,
		save_csv=True,
		save_agent=True,
		enable_wandb=True,
		save_video=True,
		wandb_silent=True,
		wandb_project='none',
		wandb_entity='none',
		task='dog-run',
		task_title='Dog Run',
		exp_name='smoke',
		seed=1,
		steps=10,
		obs_shape={'state': (4,)},
		action_dim=2,
		multitask=False,
	)
	cfg.get = lambda k, default=None: getattr(cfg, k, default)
	return cfg


def test_logger_rank_gate():
	with tempfile.TemporaryDirectory() as tmp_dir:
		rank1_dir = Path(tmp_dir) / 'rank1'
		cfg_rank1 = _cfg(rank1_dir, global_rank=1)
		logger_rank1 = Logger(cfg_rank1)
		assert logger_rank1._enabled is False
		assert not rank1_dir.exists()
		assert cfg_rank1.enable_wandb is False
		assert cfg_rank1.save_agent is False

		rank0_dir = Path(tmp_dir) / 'rank0'
		cfg_rank0 = _cfg(rank0_dir, global_rank=0)
		logger_rank0 = Logger(cfg_rank0)
		assert logger_rank0._enabled is True
		assert rank0_dir.exists()
		assert (rank0_dir / 'models').exists()


if __name__ == '__main__':
	test_logger_rank_gate()
	print('logger_rank0_smoke: all checks passed')
