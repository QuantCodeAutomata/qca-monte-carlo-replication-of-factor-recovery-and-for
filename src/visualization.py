"""
Visualization utilities for Monte Carlo and empirical results.

Uses matplotlib and seaborn for all plots.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


# Style settings
sns.set_theme(style="whitegrid", palette="colorblind")
FIGURE_DPI = 150
RESULTS_DIR = Path("results")


def plot_r2_by_scenario(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of R^2_{Fhat,F} across Monte Carlo scenarios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Monte Carlo results table.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    plt.Figure
    """
    from typing import Optional

    fig, ax = plt.subplots(figsize=(12, 5))

    labels = results_df["label"].tolist()
    r2_vals = results_df["R2_Fhat_F"].tolist()

    colors = sns.color_palette("colorblind", len(labels))
    bars = ax.bar(labels, r2_vals, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel(r"$R^2_{\hat{F},F}$", fontsize=12)
    ax.set_title(
        r"Factor Recovery Statistic $R^2_{\hat{F},F}$ by Monte Carlo Scenario",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Perfect recovery")
    ax.legend(fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, r2_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_s2_heatmap(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of S^2_{yhat,ytilde} across scenarios and selection rules.

    Parameters
    ----------
    results_df : pd.DataFrame
        Monte Carlo results table.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    plt.Figure
    """
    from typing import Optional

    rules = ["true", "ic1", "ic2", "ic3", "aic", "bic"]
    rule_labels = ["True r", "ICp1", "ICp2", "ICp3", "AIC", "BIC"]

    s2_cols = [f"S2_{r}" for r in rules]
    available_cols = [c for c in s2_cols if c in results_df.columns]
    available_labels = [rule_labels[i] for i, c in enumerate(s2_cols) if c in results_df.columns]

    s2_matrix = results_df.set_index("label")[available_cols].values

    fig, ax = plt.subplots(figsize=(10, max(6, len(results_df) * 0.4)))

    sns.heatmap(
        s2_matrix,
        ax=ax,
        xticklabels=available_labels,
        yticklabels=results_df["label"].tolist(),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": r"$S^2_{\hat{y},\tilde{y}}$"},
    )

    ax.set_title(
        r"Forecast Closeness $S^2_{\hat{y},\tilde{y}}$ by Scenario and Selection Rule",
        fontsize=13,
    )
    ax.set_xlabel("Factor Selection Rule", fontsize=12)
    ax.set_ylabel("Scenario", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_factor_selection_counts(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of mean selected factor counts across scenarios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Monte Carlo results table.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    plt.Figure
    """
    from typing import Optional

    rules = ["ic1", "ic2", "ic3", "aic", "bic"]
    rule_labels = ["ICp1", "ICp2", "ICp3", "AIC", "BIC"]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results_df))
    labels = results_df["label"].tolist()

    for rule, label in zip(rules, rule_labels):
        col = f"r_mean_{rule}"
        if col in results_df.columns:
            ax.plot(x, results_df[col].values, marker="o", label=label, linewidth=1.5)

    # True r_true line
    if "r_true" in results_df.columns:
        ax.plot(x, results_df["r_true"].values, "k--", linewidth=2, label="True r", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Mean Selected Factor Count", fontsize=12)
    ax.set_title("Mean Selected Factor Counts by Scenario and Criterion", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_empirical_forecasts(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot empirical forecast comparison: realized vs. PC and AR forecasts.

    Parameters
    ----------
    results_df : pd.DataFrame
        Expanding-window forecast results.
    metrics_df : pd.DataFrame
        Forecast metrics table.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    plt.Figure
    """
    from typing import Optional

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top panel: time series of forecasts vs. realized
    ax1 = axes[0]
    df = results_df.dropna(subset=["realized"])

    ax1.plot(df.index, df["realized"], "k-", linewidth=1.5, label="Realized", alpha=0.8)
    if "pc_ic3_forecast" in df.columns:
        ax1.plot(df.index, df["pc_ic3_forecast"], "b-", linewidth=1.2,
                 label="PC ICp3", alpha=0.8)
    if "ar_forecast" in df.columns:
        ax1.plot(df.index, df["ar_forecast"], "r--", linewidth=1.2,
                 label="AR(4)", alpha=0.7)
    if "pc_k3_forecast" in df.columns:
        ax1.plot(df.index, df["pc_k3_forecast"], "g-.", linewidth=1.2,
                 label="PC k=3", alpha=0.7)

    ax1.set_title("12-Month-Ahead Industrial Production Growth: Forecasts vs. Realized",
                  fontsize=12)
    ax1.set_ylabel("Log Growth Rate", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: relative MSE bar chart
    ax2 = axes[1]
    if not metrics_df.empty and "relative_mse" in metrics_df.columns:
        methods = metrics_df["method"].tolist()
        rel_mse = metrics_df["relative_mse"].tolist()
        colors = ["red" if m == "AR(4)" else "steelblue" for m in methods]
        bars = ax2.bar(methods, rel_mse, color=colors, edgecolor="black", linewidth=0.5)
        ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                    label="AR benchmark (=1.0)")
        ax2.set_xlabel("Method", fontsize=11)
        ax2.set_ylabel("Relative MSE (vs. AR)", fontsize=11)
        ax2.set_title("Out-of-Sample Relative MSE: PC Forecasts vs. AR Benchmark", fontsize=12)
        ax2.legend(fontsize=10)

        for bar, val in zip(bars, rel_mse):
            if not np.isnan(val):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_r2_vs_nt(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of R^2 vs. min(N,T) to illustrate asymptotic improvement.

    Parameters
    ----------
    results_df : pd.DataFrame
        Monte Carlo results table.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    plt.Figure
    """
    from typing import Optional

    fig, ax = plt.subplots(figsize=(8, 6))

    min_nt = results_df[["T", "N"]].min(axis=1)
    r2_vals = results_df["R2_Fhat_F"]

    scatter = ax.scatter(
        min_nt, r2_vals,
        c=results_df["r_true"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    for _, row in results_df.iterrows():
        ax.annotate(
            row["label"],
            (min(row["T"], row["N"]), row["R2_Fhat_F"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
        )

    plt.colorbar(scatter, ax=ax, label="True r")
    ax.set_xlabel("min(N, T)", fontsize=12)
    ax.set_ylabel(r"$R^2_{\hat{F},F}$", fontsize=12)
    ax.set_title(r"Factor Recovery $R^2_{\hat{F},F}$ vs. Sample Size", fontsize=13)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


# Fix missing Optional import at module level
from typing import Optional
