"""
Bayesian Optimization framework for OD calibration.

This module contains:
- BOResults: Dataclass for storing BO optimization results
- BayesianOptimizationLoop: Main BO loop class
- run_multiple_bo_restarts: Wrapper for multiple BO restarts
"""

from src.optimization.results import BOResults
from src.optimization.bo_loop import BayesianOptimizationLoop
from src.optimization.restarts import run_multiple_bo_restarts
from src.optimization.io import save_bo_results, load_bo_results, load_all_restarts

__all__ = [
    "BOResults",
    "BayesianOptimizationLoop",
    "run_multiple_bo_restarts",
    "save_bo_results",
    "load_bo_results",
    "load_all_restarts",
]
