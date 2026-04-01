"""Minimal smoke tests for distributed init helpers in train.py.

Run:
  python tdmpc2/tests/ddp_init_smoke.py
"""

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist

# Make imports robust when run as:
# - python tdmpc2/tests/ddp_init_smoke.py  (repo root)
# - cd tdmpc2 && python tests/ddp_init_smoke.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tdmpc2.train import _init_distributed, _cleanup_distributed


def test_single_process_no_ddp():
    cfg = SimpleNamespace(
        use_ddp=False,
        world_size=1,
        global_rank=0,
        local_rank=0,
        device='cpu',
    )
    initialized = _init_distributed(cfg)
    assert initialized is False
    assert not dist.is_initialized()


def test_single_process_ddp_gloo_file_init():
    with tempfile.NamedTemporaryFile() as tmp:
        cfg = SimpleNamespace(
            use_ddp=True,
            world_size=1,
            global_rank=0,
            local_rank=0,
            distributed_backend='gloo',
            distributed_init_method=f'file://{tmp.name}',
            distributed_timeout_sec=30,
            device='cpu',
        )
        initialized = _init_distributed(cfg)
        assert initialized is True
        assert dist.is_initialized()
        _cleanup_distributed()
        assert not dist.is_initialized()


if __name__ == '__main__':
    test_single_process_no_ddp()
    test_single_process_ddp_gloo_file_init()
    print('ddp_init_smoke: all checks passed')
