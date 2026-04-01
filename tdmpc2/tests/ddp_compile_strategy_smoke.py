"""Smoke test for DDP compile strategy resolver.

Run:
  python tdmpc2/tests/ddp_compile_strategy_smoke.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from tdmpc2.train import _resolve_compile_strategy


def test_ddp_compile_off_forces_false():
	cfg = SimpleNamespace(world_size=2, compile=True, ddp_compile_strategy='off')
	_resolve_compile_strategy(cfg)
	assert cfg.compile is False


def test_ddp_compile_on_keeps_flag():
	cfg = SimpleNamespace(world_size=2, compile=True, ddp_compile_strategy='on')
	_resolve_compile_strategy(cfg)
	assert cfg.compile is True


def test_single_process_unchanged():
	cfg = SimpleNamespace(world_size=1, compile=True, ddp_compile_strategy='off')
	_resolve_compile_strategy(cfg)
	assert cfg.compile is True


if __name__ == '__main__':
	test_ddp_compile_off_forces_false()
	test_ddp_compile_on_keeps_flag()
	test_single_process_unchanged()
	print('ddp_compile_strategy_smoke: all checks passed')
