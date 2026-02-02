"""
Visualization functions for BOResults objects.

This module contains functions to visualize BO results including:
- Flow coverage plots
- Best iteration plots
- Per-iteration comparison plots
- OD comparison plots
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch.utils.transforms import unnormalize


def plot_flow_coverage_per_edge(
    df_edge_stats: pd.DataFrame,
    gt_edge_data: pd.DataFrame,
    edge_ids: List[str],
    save_path: Optional[str] = None,
):
    """
    For each edge, show the distribution of simulated flows across BO
    evaluations and the GT flow as a reference.
    """
    if df_edge_stats is None or df_edge_stats.empty:
        print("df_edge_stats is empty – nothing to plot.")
        return

    df = df_edge_stats[df_edge_stats["edge_id"].isin(edge_ids)].copy()

    edge_ids_ordered = list(edge_ids)
    edge_to_idx = {e: i for i, e in enumerate(edge_ids_ordered)}
    df["edge_idx"] = df["edge_id"].map(edge_to_idx)

    gt_flows = (
        gt_edge_data.set_index("edge_id")
        .loc[edge_ids_ordered, "interval_nVehContrib"]
    )

    x_pos = np.arange(len(edge_ids_ordered))

    plt.figure(figsize=(12, 6))

    # All simulated flows (blue dots)
    plt.scatter(
        df["edge_idx"],
        df["interval_nVehContrib"],
        alpha=0.4,
        label="Simulated flows",
    )

    # GT flows (red stars)
    plt.scatter(
        x_pos,
        gt_flows.values,
        color="red",
        marker="*",
        s=150,
        label="GT flow",
    )

    plt.xticks(x_pos, edge_ids_ordered, rotation=45)
    plt.xlabel("Edge ID")
    plt.ylabel("Flow (interval_nVehContrib)")
    plt.title("Coverage of simulated flows vs GT per edge")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved flow coverage plot to {save_path}")

    plt.show()


def plot_best_iteration_like_prof(
    results,
    gt_edge_data,
    edge_ids,
    gt_od_vals,
    bounds,
    save_dir=None,
):
    """
    Reproduce professor-style plots for the BEST BO iteration:
      1) GT edge counts vs Simulated edge counts (scatter)
      2) Best OD vs GT OD (bar chart)

    Parameters
    ----------
    results : BOResults
        Output of bo_loop.run_bo_loop(...)
    gt_edge_data : pd.DataFrame
        Ground-truth edge stats (with columns ['edge_id', 'interval_nVehContrib'])
    edge_ids : list of str
        List of edges that are calibrated.
    gt_od_vals : np.ndarray
        Ground truth OD vector (same order as in your model).
    bounds : torch.Tensor
        Bounds used for unnormalize (2 x d tensor).
    save_dir : str or None
        If provided, figures are also saved to this directory.
    """
    df_edge_stats = results.df_edge_stats
    if df_edge_stats is None or df_edge_stats.empty:
        print("WARNING: results.df_edge_stats is empty – nothing to plot.")
        return

    #Best iteration index
    best_iter = int(results.best_idx)
    print(f"Best BO iteration index: {best_iter}")

    # Select simulated flows for that iteration
    curr_edge_stats = df_edge_stats[df_edge_stats["bo_iteration"] == best_iter]
    # Merge with GT to have both columns side by side
    df1b = gt_edge_data.merge(
        curr_edge_stats,
        on="edge_id",
        how="left",
        suffixes=("_gt", "_bo"),
    )

    # In case some edges have no sim value (shouldn't happen, but be safe)
    df1b["interval_nVehContrib_bo"] = df1b["interval_nVehContrib_bo"].fillna(0.0)

    #  Plot GT vs Sim edge counts
    plt.figure(figsize=(6, 5))
    max_val = np.max(
        [
            df1b["interval_nVehContrib_gt"].max(),
            df1b["interval_nVehContrib_bo"].max(),
        ]
    )
    vec = np.linspace(0, max_val, 100)
    plt.plot(vec, vec, "r-", label="Perfect fit")
    plt.plot(
        df1b["interval_nVehContrib_gt"],
        df1b["interval_nVehContrib_bo"],
        "x",
        label="Edges",
    )
    plt.xlabel("GT edge counts")
    plt.ylabel("Simulated edge counts")
    plt.title(f"Best BO iteration: {best_iter}; S = {results.best_S:.6f}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(
            Path(save_dir) / f"best_iter_{best_iter}_edge_counts.png",
            dpi=300,
            bbox_inches="tight",
        )

    # 3) Plot best OD vs GT OD
    plt.figure(figsize=(6, 5))

    # Unnormalize the best OD (same index as best S)
    best_od_real = (
        unnormalize(results.all_X[best_iter : best_iter + 1], bounds)
        .squeeze()
        .cpu()
        .numpy()
    )

    x = np.arange(len(best_od_real))
    width = 0.35
    plt.bar(x, best_od_real, width, label="BO", alpha=0.8)
    plt.bar(x + width, gt_od_vals, width, label="GT", alpha=0.8)

    plt.xlabel("OD pair")
    plt.ylabel("Demand")
    plt.title(f"Best BO iteration: {best_iter}")
    plt.xticks(x + width / 2, [f"OD{i}" for i in range(len(best_od_real))])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    if save_dir is not None:
        plt.savefig(
            Path(save_dir) / f"best_iter_{best_iter}_od_vs_gt.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def _plot_iteration_like_prof_internal(
    results,
    gt_edge_data: pd.DataFrame,
    edge_ids,
    gt_od_vals,
    bounds: torch.Tensor,
    eval_idx: int,
    save_dir: str | None = None,
):
    """
    Internal helper: make the 2-panel figure (GT vs sim flows, and OD bars)
    for a given *evaluation index*.

    eval_idx is the global evaluation index:
      0..n_init-1  -> initial samples
      n_init..end  -> BO iterations
    """

    if results.df_edge_stats is None or results.df_edge_stats.empty:
        raise ValueError("results.df_edge_stats is empty – did you run BO with recording?")

    # 1) Edge counts scatter
    df_edge_stats = results.df_edge_stats.copy()

    curr_edge_stats = df_edge_stats[df_edge_stats["bo_iteration"] == eval_idx]
    if curr_edge_stats.empty:
        print(f"[plot] No edge stats for eval_idx {eval_idx} – skipping.")
        return

    df_merged = gt_edge_data.merge(
        curr_edge_stats, on="edge_id", how="left",
        suffixes=("_gt", "_bo")
    )

    # loss just for info
    gt_flows = df_merged["interval_nVehContrib_gt"].to_numpy()
    sim_flows = df_merged["interval_nVehContrib_bo"].to_numpy()
    weights = gt_flows / gt_flows.sum()
    per_edge_sq_rel = ((sim_flows - gt_flows) / (gt_flows + 1e-9)) ** 2
    S_val = float((per_edge_sq_rel * weights).sum())

    # 2) OD vector at that eval
    # all_X is normalized; row index == evaluation index
    best_X_norm = results.all_X[eval_idx].unsqueeze(0)
    best_X_real = unnormalize(best_X_norm, bounds).squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    # --- Top: GT vs simulated edge flows ---
    ax = axes[0]
    max_val = max(gt_flows.max(), sim_flows.max())
    vec = np.linspace(0, max_val, 100)
    ax.plot(vec, vec, "r-", label="Perfect fit")
    ax.plot(gt_flows, sim_flows, "bx", label="Edges")
    ax.set_xlabel("GT edge counts")
    ax.set_ylabel("Simulated edge counts")
    ax.set_title(f"BO eval index: {eval_idx}; S = {S_val:.6f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    width = 0.35
    x_pos = np.arange(len(best_X_real))

    ax2.bar(x_pos - width/2, best_X_real, width, label="BO", color="tab:blue")
    ax2.bar(x_pos + width/2, gt_od_vals, width, label="GT", color="tab:orange")
    ax2.set_xlabel("OD pair")
    ax2.set_ylabel("Demand")
    ax2.set_title(f"BO eval index: {eval_idx}")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i*0.5:.1f}" for i in range(len(best_X_real))])  # optional
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"bo_eval_{eval_idx:02d}_prof_style.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved {fig_path}")

    plt.show()


def plot_iteration_like_prof(
    results,
    gt_edge_data: pd.DataFrame,
    edge_ids,
    gt_od_vals,
    bounds: torch.Tensor,
    eval_idx: int,
    save_dir: str | None = None,
):
    """
    Public helper: plot ONE evaluation (initial or BO) in the same style
    as your professor: GT vs sim flows + OD vector vs GT.
    """
    _plot_iteration_like_prof_internal(
        results=results,
        gt_edge_data=gt_edge_data,
        edge_ids=edge_ids,
        gt_od_vals=gt_od_vals,
        bounds=bounds,
        eval_idx=eval_idx,
        save_dir=save_dir,
    )


def plot_all_iterations_like_prof(
    results,
    gt_edge_data: pd.DataFrame,
    edge_ids,
    gt_od_vals,
    bounds: torch.Tensor,
    save_dir: str | None = None,
):
    """
    Plot ONLY the BO iterations (not the initial Sobol samples).
    BO iterations start at: results.iteration_start
    """
    n_evals = results.all_X.shape[0]
    start = results.iteration_start           # skip initial samples

    print(f"Plotting BO iterations {start}..{n_evals-1} (total {n_evals-start})")

    for eval_idx in range(start, n_evals):
        _plot_iteration_like_prof_internal(
            results=results,
            gt_edge_data=gt_edge_data,
            edge_ids=edge_ids,
            gt_od_vals=gt_od_vals,
            bounds=bounds,
            eval_idx=eval_idx,
            save_dir=save_dir,
        )


def plot_od_comparison_per_iteration(
    results,
    gt_od_vals,
    bounds,
    save_dir=None
):
    """
    Plot BO vs GT OD demands for EVERY BO iteration.
    Shows ONLY the bar comparison (no edge scatter plot).
    """
    all_X = results.all_X   # normalized ODs
    it_start = results.iteration_start      # START OF BO ITERATIONS
    n_total = all_X.shape[0]

    # Create directory if needed
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Plotting BO iterations {it_start}..{n_total-1}")

    for i in range(it_start, n_total):
        # Extract OD vector (normalized)
        x_norm = all_X[i].unsqueeze(0)

        # Unnormalize to real OD values
        x_real = unnormalize(x_norm, bounds).cpu().numpy().flatten()

        plt.figure(figsize=(8,5))
        width = 0.35
        idx = np.arange(len(gt_od_vals))

        plt.bar(idx, x_real, width, label="BO", color="#1f77b4")
        plt.bar(idx + width, gt_od_vals, width, label="GT", color="#d95f02")

        plt.xlabel("OD pair")
        plt.ylabel("Demand")
        plt.title(f"BO iteration {i - it_start}")  # cleaner numbering
        plt.xticks(idx + width/2, [f"{j}" for j in range(len(gt_od_vals))])
        plt.legend()

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/bo_od_iter_{i - it_start}.png",
                dpi=200,
                bbox_inches="tight"
            )

        plt.show()


def plot_flow_histogram_per_iteration(
    results,
    gt_edge_data: pd.DataFrame,
    edge_ids: List[str],
    eval_idx: int,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Plot histogram comparing GT vs simulated flows for each link at a specific iteration.
    
    Creates a grid of histograms, one per link, showing GT (red) and simulated (blue) flows side-by-side.
    
    Parameters
    ----------
    results : BOResults
        Output of bo_loop.run_bo_loop(...)
    gt_edge_data : pd.DataFrame
        Ground-truth edge stats (with columns ['edge_id', 'interval_nVehContrib'])
    edge_ids : list of str
        List of edges to plot
    eval_idx : int
        Evaluation index (iteration) to plot
    save_path : str or None
        If provided, figure is saved to this path
    figsize : tuple
        Figure size (width, height)
    """
    if results.df_edge_stats is None or results.df_edge_stats.empty:
        raise ValueError("results.df_edge_stats is empty – did you run BO with recording?")
    
    df_edge_stats = results.df_edge_stats.copy()
    curr_edge_stats = df_edge_stats[df_edge_stats["bo_iteration"] == eval_idx]
    
    if curr_edge_stats.empty:
        print(f"[plot] No edge stats for eval_idx {eval_idx} – skipping.")
        return
    
    # Merge GT and simulated data
    df_merged = gt_edge_data.merge(
        curr_edge_stats, on="edge_id", how="inner",
        suffixes=("_gt", "_sim")
    )
    
    # Filter to only edges we want to plot
    df_merged = df_merged[df_merged["edge_id"].isin(edge_ids)].copy()
    
    if df_merged.empty:
        print(f"[plot] No matching edges found for eval_idx {eval_idx}")
        return
    
    # Sort by edge_id for consistent ordering
    df_merged = df_merged.sort_values("edge_id").reset_index(drop=True)
    
    # Calculate grid dimensions
    n_edges = len(df_merged)
    n_cols = min(4, n_edges)  # Max 4 columns
    n_rows = int(np.ceil(n_edges / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_edges == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get GT and simulated flows
    gt_flows = df_merged["interval_nVehContrib_gt"].values
    sim_flows = df_merged["interval_nVehContrib_sim"].values
    edge_names = df_merged["edge_id"].values
    
    # Determine flow range for consistent scaling
    max_flow = max(gt_flows.max(), sim_flows.max())
    min_flow = min(gt_flows.min(), sim_flows.min())
    flow_range = max_flow - min_flow if max_flow > min_flow else max_flow
    
    for i, edge_id in enumerate(edge_names):
        ax = axes[i]
        
        gt_val = gt_flows[i]
        sim_val = sim_flows[i]
        
        # Plot side-by-side bars
        x_pos = np.array([0, 1])  # GT at 0, Simulated at 1
        width = 0.35
        bars = ax.bar(x_pos, [gt_val, sim_val], width=width, 
                     color=['red', 'blue'], alpha=0.7, 
                     label=['GT', 'Simulated'] if i == 0 else ['', ''])
        
        # Add value labels on bars
        ax.text(x_pos[0], gt_val, f'{gt_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(x_pos[1], sim_val, f'{sim_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Calculate error percentage
        if gt_val > 0:
            error_pct = abs(sim_val - gt_val) / gt_val * 100
            ax.text(0.5, max(gt_val, sim_val) * 1.1, f'Error: {error_pct:.1f}%', 
                   ha='center', fontsize=9, color='darkgreen' if error_pct < 10 else 'darkred')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['GT', 'Simulated'], fontsize=9)
        ax.set_ylabel('Flow', fontsize=10)
        ax.set_title(f'{edge_id}', fontsize=11, fontweight='bold')
        ax.set_ylim([0, max(gt_val, sim_val) * 1.25])
        ax.grid(True, alpha=0.3, axis='y')
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
    
    # Hide unused subplots
    for i in range(n_edges, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'GT vs Simulated Flows per Link - Iteration {eval_idx}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved histogram plot to {save_path}")
    
    plt.show()


def plot_flow_histogram_all_iterations(
    results,
    gt_edge_data: pd.DataFrame,
    edge_ids: List[str],
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Plot histogram comparing GT vs simulated flows for each link at ALL BO iterations.
    
    Creates one plot file per iteration showing histograms for all links.
    
    Parameters
    ----------
    results : BOResults
        Output of bo_loop.run_bo_loop(...)
    gt_edge_data : pd.DataFrame
        Ground-truth edge stats (with columns ['edge_id', 'interval_nVehContrib'])
    edge_ids : list of str
        List of edges to plot
    save_dir : str or None
        If provided, figures are saved to this directory
    figsize : tuple
        Figure size (width, height)
    """
    if results.df_edge_stats is None or results.df_edge_stats.empty:
        raise ValueError("results.df_edge_stats is empty – did you run BO with recording?")
    
    df_edge_stats = results.df_edge_stats.copy()
    iteration_start = results.iteration_start
    n_evals = results.all_X.shape[0]
    
    print(f"Plotting histograms for BO iterations {iteration_start}..{n_evals-1}")
    
    for eval_idx in range(iteration_start, n_evals):
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"flow_histogram_iter_{eval_idx:02d}.png"
        
        plot_flow_histogram_per_iteration(
            results=results,
            gt_edge_data=gt_edge_data,
            edge_ids=edge_ids,
            eval_idx=eval_idx,
            save_path=save_path,
            figsize=figsize,
        )
