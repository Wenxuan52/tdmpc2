#!/usr/bin/env python3
"""Visualize trained mt80 task embeddings from a checkpoint with UMAP/t-SNE.

This version draws a wider figure and only annotates clustered / nearby tasks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from adjustText import adjust_text

try:
	# Works when launched as a module: `python -m tdmpc2.visual_embedding`
	from tdmpc2.common import TASK_SET
except ModuleNotFoundError:
	# Works when launched as a script path: `python tdmpc2/visual_embedding.py`
	from common import TASK_SET


EMBED_SAVE_DIR = Path("/media/datasets/cheliu21/cxy_worldmodel/embeddings/")
EMBED_SAVE_PATH = EMBED_SAVE_DIR / "task_embedding_trained_mt80.pt"
FIG_SAVE_PATH = Path("figures/embeddings_small.png")

DMCONTROL_COLOR = "#5da7df"
METAWORLD_COLOR = "#d64a4b"

REDUCTION_METHOD = "tsne"  # choose from {"umap", "tsne"}
SEED = 999

CHECKPOINT_PATH = Path(
	"/media/datasets/cheliu21/cxy_worldmodel/checkpoint/"
	"mt80_317M_8gpu_cpu19_reserve23_20260403_024905/gpu4/mt80/321/final.pt"
)


# Only annotate tasks that form visually meaningful nearby clusters.
# All points are still drawn, but isolated / less informative tasks are not labeled.
CLUSTERED_LABEL_TASKS = {
	# DMControl locomotion clusters
	"hopper-stand",
	"hopper-hop",
	"hopper-hop-backwards",

	"walker-stand",
	"walker-walk",
	"walker-run",
	"walker-walk-backwards",
	"walker-run-backwards",

	"cheetah-run",
	"cheetah-run-front",
	"cheetah-run-back",
	"cheetah-run-backwards",
	"cheetah-jump",

	# DMControl control / manipulation-like clusters
	"cartpole-balance",
	"cartpole-balance-sparse",
	"cartpole-swingup",
	"cartpole-swingup-sparse",

	"pendulum-swingup",
	"pendulum-spin",
	"acrobot-swingup",

	"reacher-easy",
	"reacher-hard",
	"reacher-three-easy",
	"reacher-three-hard",

	"finger-spin",
	"finger-turn-easy",
	"finger-turn-hard",

	"cup-catch",
	"cup-spin",

	# MetaWorld button / coffee clusters
	"mw-button-press",
	"mw-button-press-wall",
	"mw-button-press-topdown",
	"mw-button-press-topdown-wall",
	"mw-coffee-button",
	"mw-coffee-push",
	"mw-coffee-pull",

	# MetaWorld door / drawer / window clusters
	"mw-door-lock",
	"mw-door-unlock",
	"mw-door-close",
	"mw-drawer-open",
	"mw-drawer-close",
	"mw-window-open",
	"mw-window-close",

	# MetaWorld plate / faucet nearby clusters
	"mw-faucet-open",
	"mw-faucet-close",
	"mw-plate-slide",
	"mw-plate-slide-side",
	"mw-plate-slide-back",
	"mw-plate-slide-back-side",

	# MetaWorld pick / place / shelf / peg clusters
	"mw-pick-place",
	"mw-pick-place-wall",
	"mw-shelf-place",
	"mw-peg-insert-side",
	"mw-peg-unplug-side",
	"mw-pick-out-of-hole",

	# MetaWorld handle / lever clusters
	"mw-handle-press",
	"mw-handle-press-side",
	"mw-handle-pull",
	"mw-handle-pull-side",
	"mw-lever-pull",

	# MetaWorld push / reach / sweep / stick clusters
	"mw-push",
	"mw-push-wall",
	"mw-push-back",
	"mw-reach",
	"mw-reach-wall",
	"mw-sweep-into",
	"mw-stick-push",
	"mw-stick-pull",

	# Other nearby MetaWorld points in dense regions
	"mw-basketball",
	"mw-soccer",
	"mw-box-close",
	"mw-hand-insert",
}


def _title_from_dash_name(name: str) -> str:
	"""
	Convert dash-separated task name to title-style label.

	Examples:
		finger-turn-hard -> Finger Turn Hard
		cartpole-balance-sparse -> Cartpole Balance Sparse
	"""
	words = [word for word in name.split("-") if word]
	return " ".join(word[:1].upper() + word[1:] for word in words)


def _format_task_label(task_name: str) -> str:
	"""
	Format task label for visualization.

	DMControl:
		finger-turn-hard -> Finger Turn Hard

	MetaWorld:
		mw-pick-place-wall -> Pick Place Wall
	"""
	if task_name.startswith("mw-"):
		task_name = task_name[len("mw-"):]
	return _title_from_dash_name(task_name)


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
		raise ValueError(
			f"Expected {len(tasks)} tasks for mt80, got weight shape {tuple(weight.shape)}."
		)

	return tasks, weight


def _reduce_embedding(weight: np.ndarray, method: str, seed: int) -> np.ndarray:
	method = method.lower()

	if method == "tsne":
		reducer = TSNE(
			n_components=2,
			random_state=seed,
			init="pca",
			learning_rate="auto",
			perplexity=20,
		)
		return reducer.fit_transform(weight)

	if method == "umap":
		try:
			import umap  # type: ignore
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError(
				"UMAP is selected but `umap-learn` is not installed. "
				"Please install it via `pip install umap-learn`."
			) from exc

		reducer = umap.UMAP(
			n_components=2,
			random_state=seed,
			n_neighbors=12,
			min_dist=0.35,
		)
		return reducer.fit_transform(weight)

	raise ValueError(f"Unsupported method '{method}'. Choose from ['umap', 'tsne'].")


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

	# Wider figure for main / appendix presentation.
	fig, ax = plt.subplots(figsize=(24, 18))

	# Add margin so labels have room to move.
	x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
	y_min, y_max = xy[:, 1].min(), xy[:, 1].max()

	x_range = x_max - x_min
	y_range = y_max - y_min

	x_pad = x_range * 0.14
	y_pad = y_range * 0.12

	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)

	# Larger points.
	marker_size = 3200

	ax.scatter(
		dm_xy[:, 0],
		dm_xy[:, 1],
		s=marker_size,
		marker="s",
		c=DMCONTROL_COLOR,
		edgecolors="white",
		linewidths=2.5,
		alpha=0.65,
	)

	ax.scatter(
		mw_xy[:, 0],
		mw_xy[:, 1],
		s=marker_size,
		marker="o",
		c=METAWORLD_COLOR,
		edgecolors="white",
		linewidths=2.5,
		alpha=0.65,
	)

	texts = []
	label_x = []
	label_y = []

	for idx, task_name in enumerate(tasks):
		if task_name not in CLUSTERED_LABEL_TASKS:
			continue

		x, y = xy[idx]
		label = _format_task_label(task_name)

		# texts.append(
		# 	ax.text(
		# 		x,
		# 		y,
		# 		label,
		# 		fontsize=22,
		# 		fontweight="bold",
		# 		color="#1f2430",
		# 		alpha=0.88,
		# 		ha="center",
		# 		va="bottom",
		# 	)
		# )

		label_x.append(x)
		label_y.append(y)

	label_x = np.asarray(label_x)
	label_y = np.asarray(label_y)

	adjust_text(
		texts,

		# Avoid all 80 task points.
		x=xy[:, 0],
		y=xy[:, 1],

		# Arrows only point back to labeled tasks.
		target_x=label_x,
		target_y=label_y,

		ax=ax,

		# Keep labels relatively close to their own points.
		expand_text=(1.08, 1.15),
		expand_points=(1.05, 1.12),
		force_text=(0.28, 0.36),
		force_points=(0.10, 0.18),

		lim=2500,
		min_arrow_len=0,

		arrowprops=dict(
			arrowstyle="->",
			color="#222222",
			lw=1.2,
			alpha=0.65,
			shrinkA=6,
			shrinkB=0,
			mutation_scale=10,
		),
	)

	ax.set_xticks([])
	ax.set_yticks([])

	for spine in ax.spines.values():
		spine.set_visible(False)

	FIG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

	plt.tight_layout(pad=1.0)
	plt.savefig(FIG_SAVE_PATH, bbox_inches="tight", pad_inches=0.1)
	plt.close()

	print(f"[visual_embedding] Saved {REDUCTION_METHOD.upper()} figure to: {FIG_SAVE_PATH}")


if __name__ == "__main__":
	main()