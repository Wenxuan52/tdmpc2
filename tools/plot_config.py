from __future__ import annotations
from pathlib import Path


def load_plot_config(path: Path | None = None) -> dict[str, float | str]:
    cfg_path = path or (Path(__file__).resolve().parent / "plot_config.yaml")
    cfg: dict[str, float | str] = {}
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        v = val.strip().strip('"').strip("'")
        try:
            cfg[key.strip()] = float(v)
        except ValueError:
            cfg[key.strip()] = v
    return cfg
