from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_profiles(data: pd.DataFrame, profiles: np.ndarray, scales: np.ndarray, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    types = list(dict.fromkeys(data["candidate_type"]))
    palette = sns.color_palette("tab10", n_colors=len(types))
    for color, kind in zip(palette, types, strict=True):
        mask = data["candidate_type"].to_numpy() == kind
        values = profiles[mask]
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / np.sqrt(max(values.shape[0], 1))
        ax.plot(scales, mean, label=kind.replace("_", " "), color=color)
        ax.fill_between(scales, mean - 1.96 * se, mean + 1.96 * se, alpha=0.15, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Kernel scale")
    ax.set_ylabel("Marginal magnitude gain")
    ax.set_title("Magnitude profile by semantic relationship (mean ± 95% CI)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    ordered = results.sort_values("roc_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(8, 0.35 * len(ordered))))
    ax.barh(ordered["model"], ordered["roc_auc_mean"], xerr=ordered["roc_auc_std"], alpha=0.85)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Grouped-CV ROC AUC (mean ± fold SD)")
    ax.set_title("Does the magnitude profile add information beyond similarity?")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
