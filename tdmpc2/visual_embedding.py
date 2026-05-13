#!/usr/bin/env python3
"""Visualize trained mt80 task embeddings from a checkpoint with UMAP/t-SNE."""

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
EMBED_SAVE_PATH = EMBED_SAVE_DIR / "task_embedding_trained_mt80.pt"
FIG_SAVE_PATH = Path("figures/embeddings.png")

DMCONTROL_COLOR = "#5da7df"
METAWORLD_COLOR = "#d64a4b"
REDUCTION_METHOD = "umap"  # choose from {"umap", "tsne"}
SEED = 0
CHECKPOINT_PATH = Path(
	"/media/datasets/cheliu21/cxy_worldmodel/checkpoint/"
	"mt80_317M_8gpu_cpu19_reserve23_20260403_024905/gpu4/mt80/321/final.pt"
)


def _load_trained_mt80_embedding(ckpt_path: Path) -> tuple[list[str], torch.Tensor]:
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	ckpt = torch.load(ckpt_path, map_location="cpu")
	state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
	if not isinstance(state_dict, dict):
		raise TypeError(f"Unexpected checkpoint format at {ckpt_path}.")

	candidates = [
		"_task_emb.weight",
		"model._task_emb.weight",
		"agent.model._task_emb.weight",
	]
	weight = None
	for key in candidates:
		if key in state_dict:
			weight = state_dict[key]
			break
	if weight is None:
		for key, value in state_dict.items():
			if key.endswith("_task_emb.weight"):
				weight = value
				break
	if weight is None:
		raise KeyError(
			"Could not find task embedding weight in checkpoint. "
			"Tried keys: _task_emb.weight / model._task_emb.weight / agent.model._task_emb.weight"
		)

	tasks = TASK_SET["mt80"]
	weight = weight.detach().cpu()
	if weight.shape[0] != len(tasks):
		raise ValueError(f"Expected {len(tasks)} tasks for mt80, got weight shape {tuple(weight.shape)}.")
	return tasks, weight


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
	tasks, weight = _load_trained_mt80_embedding(CHECKPOINT_PATH)
	EMBED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"task": "mt80",
			"seed": SEED,
			"tasks": tasks,
			"checkpoint_path": str(CHECKPOINT_PATH),
			"embedding_weight": weight,
		},
		EMBED_SAVE_PATH,
	)
	print(f"[visual_embedding] Saved trained mt80 embedding matrix to: {EMBED_SAVE_PATH}")

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
