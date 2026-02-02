"""
Visualization functions for BO results.

This module contains:
- bo_results.py: Functions that work with BOResults objects (in-memory)
- convergence.py: Functions that read CSV files for post-hoc analysis
"""

from src.visualization.bo_results import (
    plot_flow_coverage_per_edge,
    plot_best_iteration_like_prof,
    plot_iteration_like_prof,
    plot_all_iterations_like_prof,
    plot_od_comparison_per_iteration,
    plot_flow_histogram_per_iteration,
    plot_flow_histogram_all_iterations,
)

from src.visualization.convergence import (
    plot_fitGT,
    plot_two_convergence,
    plot_restarts_with_mean,
    plot_restarts_mean_only,
    plot_restarts_mean_with_ci,
    plot_convergence,
)

__all__ = [
    # BOResults-based plots (in-memory)
    "plot_flow_coverage_per_edge",
    "plot_best_iteration_like_prof",
    "plot_iteration_like_prof",
    "plot_all_iterations_like_prof",
    "plot_od_comparison_per_iteration",
    "plot_flow_histogram_per_iteration",
    "plot_flow_histogram_all_iterations",
    # CSV-based plots (post-hoc analysis)
    "plot_fitGT",
    "plot_two_convergence",
    "plot_restarts_with_mean",
    "plot_restarts_mean_only",
    "plot_restarts_mean_with_ci",
    "plot_convergence",
]
