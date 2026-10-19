# -*- coding: utf-8 -*-
"""Headless (Agg) matplotlib helpers for training curves, evaluation
dashboards, trajectory maps and dataset analysis.  All functions save a PNG and
return its path so the pipeline produces reviewable artifacts on the server.
"""

from __future__ import annotations
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for r in rows:
        for k, v in r.items():
            if v == "" or v is None:
                continue
            cols.setdefault(k, []).append(float(v))
    return cols


def plot_training_curves(metrics_csv: str, out_path: str, x: str = "step",
                         ys: list[str] | None = None, title: str = "Training"):
    cols = _read_csv(metrics_csv)
    ys = ys or [k for k in cols if k != x]
    ys = [y for y in ys if y in cols]
    n = len(ys)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    for ax, y in zip(axes[0], ys):
        ax.plot(cols[x][:len(cols[y])], cols[y], lw=1.6, color="#2b6cb0")
        ax.set_title(y)
        ax.set_xlabel(x)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def bar_compare(data: dict, metric_keys: list[str], out_path: str,
                title: str = "Comparison"):
    """data: {method_name: {metric: value}}."""
    methods = list(data.keys())
    n = len(metric_keys)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    colors = ["#a0aec0", "#3182ce", "#2f855a", "#c53030"]
    for ax, mk in zip(axes[0], metric_keys):
        vals = [data[m].get(mk, 0.0) for m in methods]
        bars = ax.bar(methods, vals, color=colors[:len(methods)])
        ax.set_title(mk)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                    ha="center", va="bottom", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_trajectories(trajs, walls, goal, start, out_path,
                      title="Closed-loop trajectories"):
    """trajs: list of [T,2] arrays; walls: [W,4] (cx,cy,hx,hy)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for w in walls:
        cx, cy, hx, hy = w
        ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                   color="#4a5568", alpha=0.7))
    for t in trajs:
        t = np.asarray(t)
        ax.plot(t[:, 0], t[:, 1], lw=1.2, alpha=0.8)
        ax.plot(t[0, 0], t[0, 1], "o", color="green", ms=4)
        ax.plot(t[-1, 0], t[-1, 1], "x", color="red", ms=6)
    ax.plot(goal[0], goal[1], "*", color="gold", ms=20, markeredgecolor="k")
    ax.plot(start[0], start[1], "s", color="green", ms=10, alpha=0.5)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def hist_grid(arrays: dict, out_path: str, title="Dataset analysis", bins=40):
    """arrays: {name: 1d np array}."""
    keys = list(arrays.keys())
    n = len(keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows),
                             squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        ax.hist(np.asarray(arrays[k]), bins=bins, color="#3182ce", alpha=0.8)
        ax.set_title(k, fontsize=10)
        ax.grid(alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def scatter_pred_true(pred, true, names, out_path, title="Prediction vs Truth"):
    pred = np.asarray(pred)
    true = np.asarray(true)
    d = pred.shape[1]
    fig, axes = plt.subplots(1, d, figsize=(4 * d, 4), squeeze=False)
    for i in range(d):
        ax = axes[0][i]
        ax.scatter(true[:, i], pred[:, i], s=4, alpha=0.3, color="#3182ce")
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")
        ax.set_title(names[i])
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
