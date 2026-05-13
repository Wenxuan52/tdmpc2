#!/usr/bin/env python3
"""Export and visualize initialized mt80 task embeddings with selectable reduction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

try:
	# Works when launched as a module: `python -m tdmpc2.visual_embedding`
	from tdmpc2.common import TASK_SET
except ModuleNotFoundError:
	# Works when launched as a script path: `python tdmpc2/visual_embedding.py`
	from common import TASK_SET


EMBED_SAVE_DIR = Path("/media/datasets/cheliu21/cxy_worldmodel/embeddings/")
EMBED_SAVE_PATH = EMBED_SAVE_DIR / "task_embedding_init_mt80.pt"
FIG_SAVE_PATH = Path("figures/embeddings.png")

DMCONTROL_COLOR = "#5da7df"
METAWORLD_COLOR = "#d64a4b"
REDUCTION_METHOD = "umap"  # choose from {"umap", "tsne"}
SEED = 0


def _build_mt80_embedding(seed: int = 0, dim: int = 96) -> tuple[list[str], torch.Tensor]:
	torch.manual_seed(seed)
	tasks = TASK_SET["mt80"]
	embedding = torch.nn.Embedding(len(tasks), dim, max_norm=1)
	return tasks, embedding.weight.detach().cpu()


def _reduce_embedding(weight: np.ndarray, method: str, seed: int) -> np.ndarray:
	method = method.lower()
	if method == "tsne":
		reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=20)
		return reducer.fit_transform(weight)
	if method == "umap":
		try:
			import umap  # type: ignore
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError(
				"UMAP is selected but `umap-learn` is not installed. "
				"Please install it via `pip install umap-learn`."
			) from exc
		reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=12, min_dist=0.35)
		return reducer.fit_transform(weight)
	raise ValueError(f"Unsupported method '{method}'. Choose from ['umap', 'tsne'].")


def _select_labels(xy: np.ndarray, tasks: list[str], max_labels: int = 18, min_dist: float = 8.0) -> list[int]:
	"""Greedy label selection to reduce overlap in plotted task names."""
	order = np.argsort(xy[:, 0] + xy[:, 1])
	selected: list[int] = []
	for idx in order:
		if len(selected) >= max_labels:
			break
		if all(np.linalg.norm(xy[idx] - xy[j]) >= min_dist for j in selected):
			selected.append(int(idx))
	# Always include a few representative tasks.
	anchors = ["walker-walk", "cheetah-run", "finger-turn-hard", "mw-pick-place", "mw-reach-wall", "mw-stick-push"]
	for name in anchors:
		if name in tasks:
			i = tasks.index(name)
			if i not in selected:
				selected.append(i)
	return selected


def main() -> None:
	tasks, weight = _build_mt80_embedding(seed=SEED, dim=96)
	EMBED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
	torch.save({"task": "mt80", "seed": SEED, "tasks": tasks, "embedding_weight": weight}, EMBED_SAVE_PATH)
	print(f"[visual_embedding] Saved mt80 embedding matrix to: {EMBED_SAVE_PATH}")

	xy = _reduce_embedding(weight.numpy(), method=REDUCTION_METHOD, seed=SEED)

	dm_xy = xy[:30]
	mw_xy = xy[30:]

	plt.figure(figsize=(14, 10), facecolor="#e8e8e8")
	ax = plt.gca()
	ax.set_facecolor("#e8e8e8")

	plt.scatter(dm_xy[:, 0], dm_xy[:, 1], s=1300, marker="s", c=DMCONTROL_COLOR, edgecolors="white", linewidths=2, alpha=0.65)
	plt.scatter(mw_xy[:, 0], mw_xy[:, 1], s=1300, marker="o", c=METAWORLD_COLOR, edgecolors="white", linewidths=2, alpha=0.65)

	for idx in _select_labels(xy, tasks):
		x, y = xy[idx]
		plt.text(x + 0.9, y + 0.9, tasks[idx], fontsize=24, fontweight="bold", color="#1f2430")

	plt.xticks([])
	plt.yticks([])
	for spine in ax.spines.values():
		spine.set_visible(False)

	FIG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(FIG_SAVE_PATH, dpi=220)
	plt.close()
	print(f"[visual_embedding] Saved {REDUCTION_METHOD.upper()} figure to: {FIG_SAVE_PATH}")


if __name__ == "__main__":
	main()
