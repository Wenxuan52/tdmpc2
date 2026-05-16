#!/usr/bin/env python3
"""Visualize trained mt80 task embeddings from a checkpoint with UMAP/t-SNE."""

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
FIG_SAVE_PATH = Path("figures/embeddings.pdf")

DMCONTROL_COLOR = "#5da7df"
METAWORLD_COLOR = "#d64a4b"
# 只在绘图阶段隐藏这两个 hopper 任务；仍保留完整 embedding 参与降维。
TASKS_TO_SKIP_IN_PLOT = {"hopper-stand", "hopper-hop-backwards"}
# TASKS_TO_SKIP_IN_PLOT = {}
REDUCTION_METHOD = "tsne"  # choose from {"umap", "tsne"}
SEED = 999
CHECKPOINT_PATH = Path(
	"/media/datasets/cheliu21/cxy_worldmodel/checkpoint/"
	"mt80_317M_8gpu_cpu19_reserve23_20260403_024905/gpu4/mt80/321/final.pt"
)


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

	# 只取消绘制 hopper-stand 和 hopper-hop-backwards；hopper-hop 仍会绘制并标注。
	visible_mask = np.array(
		[task_name not in TASKS_TO_SKIP_IN_PLOT for task_name in tasks],
		dtype=bool,
	)
	dm_mask = np.arange(len(tasks)) < 30
	mw_mask = ~dm_mask

	visible_xy = xy[visible_mask]
	dm_xy = xy[dm_mask & visible_mask]
	mw_xy = xy[mw_mask & visible_mask]

	# 图放大，但点大小和字体大小保持不变
	fig, ax = plt.subplots(figsize=(18, 28))

	# 给坐标轴留白，让 adjustText 有更多空间移动文字
	x_min, x_max = visible_xy[:, 0].min(), visible_xy[:, 0].max()
	y_min, y_max = visible_xy[:, 1].min(), visible_xy[:, 1].max()

	x_range = x_max - x_min
	y_range = y_max - y_min

	x_pad = x_range * 0.12
	y_pad = y_range * 0.05

	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)

	# 点大小保持不变：s=1300
	ax.scatter(
		dm_xy[:, 0],
		dm_xy[:, 1],
		s=2300,
		marker="s",
		c=DMCONTROL_COLOR,
		edgecolors="white",
		linewidths=2,
		alpha=0.65,
	)

	ax.scatter(
		mw_xy[:, 0],
		mw_xy[:, 1],
		s=2300,
		marker="o",
		c=METAWORLD_COLOR,
		edgecolors="white",
		linewidths=2,
		alpha=0.65,
	)

	texts = []
	text_x = []
	text_y = []

	for idx, task_name in enumerate(tasks):
		if task_name in TASKS_TO_SKIP_IN_PLOT:
			continue

		x, y = xy[idx]
		label = _format_task_label(task_name)
		text_x.append(x)
		text_y.append(y)

		texts.append(
			ax.text(
				x,
				y + 0.0,
				label,
				fontsize=16,
				fontweight="bold",
				color="#1f2430",
				alpha=0.85,
				ha="center",
				va="bottom",
			)
		)

	adjust_text(
		texts,
		x=visible_xy[:, 0],
		y=visible_xy[:, 1],
		target_x=text_x,
		target_y=text_y,
		ax=ax,

		expand_text=(1.1, 1.2),
		expand_points=(1.1, 1.2),
		force_text=(0.3, 0.4),
		force_points=(0.15, 0.25),

		# 增加迭代次数，让 adjustText 有更多机会找到更好的位置
		lim=2000,

		# 尽量让所有指示箭头都画出来
		min_arrow_len=0,

		arrowprops=dict(
			arrowstyle="->",
			color="#222222",
			lw=1.3,
			alpha=0.65,
			shrinkA=8,
			shrinkB=0,
			mutation_scale=10,
		),
	)

	ax.set_xticks([])
	ax.set_yticks([])

	for spine in ax.spines.values():
		spine.set_visible(False)

	FIG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout(pad=1.5)
	plt.savefig(FIG_SAVE_PATH, dpi=220)
	plt.close()

	print(f"[visual_embedding] Saved {REDUCTION_METHOD.upper()} figure to: {FIG_SAVE_PATH}")


if __name__ == "__main__":
	main()