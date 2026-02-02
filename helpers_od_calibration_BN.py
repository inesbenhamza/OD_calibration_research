"""
Backward-compatible wrapper for helpers_od_calibration_BN.

This module maintains backward compatibility by re-exporting all classes
and functions from the new organized structure in src/optimization/ and src/visualization/.

The code has been reorganized into:
- src/optimization/: BO loop logic (BOResults, BayesianOptimizationLoop, run_multiple_bo_restarts)
- src/visualization/: Visualization functions (plot_flow_coverage_per_edge, etc.)

This file is kept for backward compatibility. New code should import directly from:
- from src.optimization import BOResults, BayesianOptimizationLoop, run_multiple_bo_restarts
- from src.visualization import plot_flow_coverage_per_edge, plot_best_iteration_like_prof, etc.
"""

# Re-export BO optimization classes and functions
from src.optimization import (
    BOResults,
    BayesianOptimizationLoop,
    run_multiple_bo_restarts,
    save_bo_results,
    load_bo_results,
    load_all_restarts,
)

# Re-export visualization functions
from src.visualization import (
    plot_flow_coverage_per_edge,
    plot_best_iteration_like_prof,
    plot_iteration_like_prof,
    plot_all_iterations_like_prof,
    plot_od_comparison_per_iteration,
    plot_flow_histogram_per_iteration,
    plot_flow_histogram_all_iterations,
    # CSV-based convergence plots
    plot_fitGT,
    plot_two_convergence,
    plot_restarts_with_mean,
    plot_restarts_mean_only,
    plot_restarts_mean_with_ci,
    plot_convergence,
)

# Re-export data loading function (for notebook compatibility)
from src.simulation.data_loader import load_kwargs_config

__all__ = [
    # BO optimization
    "BOResults",
    "BayesianOptimizationLoop",
    "run_multiple_bo_restarts",
    "save_bo_results",
    "load_bo_results",
    "load_all_restarts",
    # Visualization (BOResults-based)
    "plot_flow_coverage_per_edge",
    "plot_best_iteration_like_prof",
    "plot_iteration_like_prof",
    "plot_all_iterations_like_prof",
    "plot_od_comparison_per_iteration",
    "plot_flow_histogram_per_iteration",
    "plot_flow_histogram_all_iterations",
    # Visualization (CSV-based convergence)
    "plot_fitGT",
    "plot_two_convergence",
    "plot_restarts_with_mean",
    "plot_restarts_mean_only",
    "plot_restarts_mean_with_ci",
    "plot_convergence",
    # Data loading
    "load_kwargs_config",
]
