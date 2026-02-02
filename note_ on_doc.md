# Codebase Documentation

## Overview

This codebase implements Bayesian Optimization (BO) for SUMO Origin-Destination (OD) matrix calibration. It supports three model types:
- **Vanilla BO**: Single GP on aggregated error
- **Independent GP**: Separate GP per edge
- **Multi-Output GP (ICM)**: Intrinsic Coregionalization Model

The codebase is organized into modular packages (`src/optimization/`, `src/visualization/`, `src/simulation/`) with backward-compatible wrappers.

---

## File Structure

### `src/` - Main Source Code

#### `src/optimization/` - BO Core Logic
- **`bo_loop.py`**: `BayesianOptimizationLoop` class - Main BO iteration loop
- **`restarts.py`**: `run_multiple_bo_restarts()` - Orchestrates multiple BO runs
- **`results.py`**: `BOResults` dataclass - Stores BO results
- **`io.py`**: `save_bo_results()`, `load_bo_results()`, `load_all_restarts()` - Save/load functionality

#### `src/models/` - Gaussian Process Models
- **`gp_models.py`**: GP model initialization utilities
- **`vanilla_bo_complete.py`**: Vanilla BO implementation
- **`ind_gp_model.py`**: Independent GP model implementation

#### `src/simulation/` - SUMO Simulation & Data Loading
- **`data_loader.py`**: Configuration loading (`load_kwargs_config()`)
- **`sumo_runner.py`**: SUMO simulation execution and XML creation
- **`evaluation.py`**: Single-run evaluation (legacy, unused in BO loop)

#### `src/utils/` - Utility Functions
- **`link_flow_analysis.py`**: Error metrics (squared, cubic, per-edge, aggregated)
- **`seed.py`**: Random seed utilities

#### `src/visualization/` - Plotting Functions
- **`bo_results.py`**: Plotting functions that work with `BOResults` objects
- **`convergence.py`**: CSV-based convergence plotting (CLI script)

### Root Level Files
- **`optimization.py`**: Main CLI entry point for running BO
- **`helpers_od_calibration_BN.py`**: Backward-compatible wrapper (re-exports from `src/`)



## Complete File Save Locations

### Base Directory Structure

All files are saved under:
```
output/{network_name}_{model_name}_{kernel}/
```

Where:
- `network_name`: e.g., `quickstart`, `berlin`
- `model_name`: `vanillabo`, `independent_gp`, `mogp`
- `kernel`: `matern-1p5`, `matern-2p5`, `rbf`

**Example**: `output/quickstart_vanillabo_matern-2p5/`

---

### 1. Ground Truth Simulation Files

**Location**: `{simulation_run_path}/ground_truth/`

**Files Created**:
```
output/{network_name}_{model_name}_{kernel}/ground_truth/
├── od.xml                          # Ground truth OD matrix XML
├── sim_edge_data.xml               # SUMO edge output (GT simulation)
├── sim_tripinfo.xml                 # SUMO trip info (if enabled)
└── sim_*.xml                        # Other SUMO output files
```

**When Created**: At the start of `optimization.py`, before initial search

**Purpose**: 
- Ground truth OD matrix (`od.xml`)
- Ground truth edge flows (`sim_edge_data.xml`) used as target for calibration
- Parsed into `gt_edge_data` DataFrame for error computation

---

### 2. Initial Search (Sobol Samples)

**Location**: `{simulation_run_path}/initial_search/sobol_{i}/`

**Files Created** (per initial sample `i`):
```
output/{network_name}_{model_name}_{kernel}/initial_search/
├── sobol_0/
│   ├── od.xml                       # OD matrix for sample 0
│   ├── sim_edge_data.xml            # SUMO edge output
│   ├── sim_tripinfo.xml             # SUMO trip info
│   └── sim_*.xml                    # Other SUMO outputs
├── sobol_1/
│   └── ...
└── sobol_{n_init_search-1}/
    └── ...
```

**When Created**: During initial search phase in `optimization.py` (lines 345-404)

**Purpose**:
- `n_init_search` Sobol quasi-random samples (default: 5)
- Each sample is evaluated via SUMO simulation
- Results used to initialize GP model before BO iterations

**Note**: If `CLEANUP_INTERMEDIATE_FILES` is enabled in config, these directories may be cleaned up after evaluation.

---

### 3. BO Iteration Files

**Location**: `{simulation_run_path}/bo_iterations/bo_iter_{iteration}/`

**Files Created** (per BO iteration):
```
output/{network_name}_{model_name}_{kernel}/bo_iterations/
├── bo_iter_0/
│   ├── od.xml                       # OD matrix proposed by acquisition function
│   ├── sim_edge_data.xml            # SUMO edge output
│   ├── sim_tripinfo.xml             # SUMO trip info
│   └── sim_*.xml                    # Other SUMO outputs
├── bo_iter_1/
│   └── ...
└── bo_iter_{n_iterations-1}/
    └── ...
```

**When Created**: During each BO iteration in `BayesianOptimizationLoop.simulate_and_evaluate()` (line 145)

**Purpose**:
- Each BO iteration proposes a new OD matrix via acquisition optimization
- OD matrix is saved as `od.xml`
- SUMO simulation is run, producing edge flow outputs
- Errors are computed and used to update the GP model

**Note**: Iteration numbering starts at 0 (first BO iteration after initial search).

---

### 4. Results Directory (Per Restart)

**Location**: `{simulation_run_path}/results/restart_{restart_idx}_seed-{seed}/`

**Files Created** (per restart):
```
output/{network_name}_{model_name}_{kernel}/results/
├── restart_1_seed-42/
│   ├── bo_results_restart_1.pkl    # Full BOResults object (pickle)
│   ├── convergence.csv              # Convergence data (iteration, best_S_model, acq_value, wall_time)
│   ├── edge_stats.csv               # Simulated flows per edge per iteration (if available)
│   ├── metadata.json                 # Metadata (best_S, best_idx, seed, model_name, kernel, etc.)
│   └── convergence.png               # Convergence plot (created by optimization.py)
├── restart_2_seed-1042/
│   └── ...
└── restart_{n_restarts}_seed-{seed}/
    └── ...
```

**When Created**: 
- Automatically by `restarts.py` after each BO restart completes (line 88)
- Also by `optimization.py` for additional CSV/plot generation (line 460)

**File Details**:

**`bo_results_restart_{i}.pkl`**:
- Complete `BOResults` object saved as pickle
- Contains: `all_X`, `all_Y_errors`, `all_S`, `best_X`, `best_S`, `convergence_curve`, `acq_values`, `wall_times`, `df_edge_stats`
- **Use**: Load with `load_bo_results()` to access all BO data without re-running

**`convergence.csv`**:
- Columns: `iteration`, `best_S_model`, `acq_value`, `wall_time`
- Format compatible with `results_visualization.py`
- **Use**: For plotting convergence curves, comparing restarts

**`edge_stats.csv`**:
- Contains simulated flows per edge per iteration
- Only saved if `df_edge_stats` is populated during BO loop
- **Use**: For flow coverage plots, edge-level analysis

**`metadata.json`**:
- Contains: `best_S`, `best_idx`, `iteration_start`, `n_evaluations`, `n_bo_iterations`, `restart_idx`, `seed`, `model_name`, `kernel`, `network_name`
- **Use**: Quick reference for experiment parameters and results

**`convergence.png`**:
- Plot of convergence curve (created by `optimization.py`)
- Shows best S vs iteration

**Seed Calculation**:
- Restart 1: seed = 42
- Restart 2: seed = 1042 (42 + 1*1000)
- Restart 3: seed = 2042 (42 + 2*1000)
- Formula: `seed = 42 + (restart_idx - 1) * 1000`

---

### 5. Overall Best Result

**Location**: `{simulation_run_path}/results/`

**Files Created**:
```
output/{network_name}_{model_name}_{kernel}/results/
├── convergence.csv                  # Best restart's convergence curve
├── bo_results.pkl                   # Best restart's BOResults object
├── metadata.json                    # Best restart's metadata
└── (restart subdirectories...)
```

**When Created**: After all restarts complete in `optimization.py` (line 495)

**Purpose**: 
- Stores the best result across all restarts
- `convergence.csv` contains the best restart's convergence curve
- `bo_results.pkl` contains the best restart's full results

---

### 6. Saved Results (Notebook Format)

**Location**: `{simulation_run_path}/saved_results/`

**Files Created** (if using notebook):
```
output/{network_name}_{model_name}_{kernel}/saved_results/
├── {model}_bo_histories.json        # All restart histories + metadata (JSON)
└── {model}_bo_histories.csv         # All restart histories (CSV format)
```

**When Created**: In notebook after BO runs complete

**Purpose**:
- JSON format: Complete restart histories with metadata
- CSV format: Easy-to-view table with columns `Restart_1`, `Restart_2`, ..., `Mean`

**Note**: This format is notebook-specific. The `optimization.py` script uses the `results/` directory structure instead.

---

### 7. Visualization Outputs

**Location**: `visualization/figures/` (from `results_visualization.py`)

**Files Created**:
```
visualization/figures/
├── convergence_{network}_{kernel}.png
├── fit_to_GT_{network}_{kernel}.png
└── (other plots...)
```

**When Created**: When running `results_visualization.py` CLI script

**Purpose**: Post-hoc analysis plots generated from saved `convergence.csv` files

---

## Loading Saved Results

### Load Single Restart

```python
from src.optimization import load_bo_results
from pathlib import Path

# Load from restart directory
results = load_bo_results(
    Path("output/quickstart_vanillabo_matern-2p5/results/restart_1_seed-42")
)

# Or load from pickle file directly
results = load_bo_results(
    Path("output/quickstart_vanillabo_matern-2p5/results/restart_1_seed-42/bo_results_restart_1.pkl")
)

# Access data
print(f"Best S: {results.best_S}")
print(f"Best OD: {results.best_X}")
print(f"Convergence: {results.convergence_curve}")
```

### Load All Restarts

```python
from src.optimization import load_all_restarts
from pathlib import Path

# Load all restarts from results directory
all_results = load_all_restarts(
    Path("output/quickstart_vanillabo_matern-2p5/results")
)

# Iterate over restarts
for i, res in enumerate(all_results):
    print(f"Restart {i+1}: Best S = {res.best_S:.6f}")
```

### Use Loaded Results for Plotting

```python
from src.visualization import plot_best_iteration_like_prof
from src.optimization import load_bo_results

# Load results
results = load_bo_results(Path("output/.../results/restart_1_seed-42"))

# Plot (requires gt_edge_data, edge_ids, gt_od_vals, bounds)
plot_best_iteration_like_prof(
    results=results,
    gt_edge_data=gt_edge_data,
    edge_ids=edge_ids,
    gt_od_vals=gt_od_vals,
    bounds=bounds,
    save_path="plot.png"
)
```

---

## Configuration System

### How Config Works

1. **Load config**: `load_kwargs_config()` in `src/simulation/data_loader.py` creates a dictionary
2. **Access config**: Use `config["key"]` or `self.config["key"]` inside classes
3. **Config keys**: Both naming conventions work:
   - UPPERCASE: `config["NUM_RESTARTS"]`, `config["RAW_SAMPLES"]`
   - Lowercase: `config["BO_num_restarts"]`, `config["BO_raw_samples"]`

### Config Dictionary Contents

**Paths**:
- `config["simulation_run_path"]` = `f"output/{network_name}_{model_name}_{kernel}"`
- `config["net_xml"]`, `config["taz2edge_xml"]`, `config["file_gt_od"]`, etc.

**Simulation settings**:
- `config["sim_start_time"]`, `config["sim_end_time"]`, `config["od_duration_sec"]`
- `config["n_init_search"]` - Number of initial Sobol samples

**BO parameters**:
- `config["NUM_RESTARTS"]` / `config["BO_num_restarts"]` - Acquisition optimization restarts
- `config["RAW_SAMPLES"]` / `config["BO_raw_samples"]` - Raw samples for acquisition
- `config["BATCH_SIZE"]` / `config["BO_batch_size"]` - Batch size
- `config["NITER"]` / `config["BO_niter"]` - Number of BO iterations

**Early stopping**:
- `config["EARLY_STOP_PATIENCE"]`, `config["EARLY_STOP_DELTA"]`, `config["EARLY_STOP_MIN_ACQ"]`

---

## Model Types

### 1. Vanilla BO (`model_name="vanillabo"`)
- **Metric**: `compute_squared_metric_all_edge()` - Returns scalar (mean relative squared error)
- **Config**: Uses `config["NUM_RESTARTS"]`, `config["RAW_SAMPLES"]` for acquisition
- **Aggregation**: Linear aggregation with uniform weight before fitting the GP
- **GP fitting**: Single GP on aggregated error

### 2. Independent GP (`model_name="independent_gp"`)
- **Metric**: `compute_squared_metric_per_edge()` or `compute_cubic_metric_per_edge()` - Returns array
- **GP**: Separate GP for each edge
- **Aggregation**: Linear aggregation with uniform or flow-proportional weights using raw output

### 3. Multi-Output GP / ICM (`model_name="mogp"` or `model_name="icm"`)
- **Metric**: `compute_squared_metric_per_edge()` or `compute_cubic_metric_per_edge()` - Returns array
- **GP**: Intrinsic Coregionalization Model (ICM) for multi-output GP
- **Standardization**: Per-edge errors are standardized before training
- **Aggregation**: Linear aggregation over standardized outputs

---

## Quick Reference

### Run BO Optimization

```bash
python optimization.py \
  --network_name quickstart \
  --model_name vanillabo \
  --kernel matern-2p5 \
  --n_iterations 30 \
  --n_restarts 5 \
  --seed 42
```

### Access Config in Code

```python
# In optimization.py
config = load_kwargs_config(base_path, model_name, sim_setup_filename, kernel)
bounds = torch.tensor([[config.get("od_bound_start", 0.0)] * dim_od, 
                       [config.get("od_bound_end", 2000.0)] * dim_od])

# In BayesianOptimizationLoop class
num_restarts = self.config.get("NUM_RESTARTS", 5)  # or config["BO_num_restarts"]
raw_samples = self.config.get("RAW_SAMPLES", 32)    # or config["BO_raw_samples"]
```

### Error Metrics
- **Vanilla BO**: `compute_squared_metric_all_edge()` → scalar
- **Independent GP**: `compute_squared_metric_per_edge()` → array
- **Independent GP (flow-weighted)**: `compute_cubic_metric_per_edge()` → array

### Save Results Manually (in Notebook)

```python
from src.optimization import save_bo_results
from pathlib import Path

save_bo_results(
    results=results,
    save_dir=Path(config['simulation_run_path']) / 'results' / 'restart_1_seed-42',
    restart_idx=1,
    seed=42,
    metadata={'model_name': 'vanillabo', 'kernel': 'matern-2p5'}
)
```

---

## Directory Structure Summary

```
output/{network_name}_{model_name}_{kernel}/
├── ground_truth/                    # GT simulation files
│   ├── od.xml
│   └── sim_*.xml
│
├── initial_search/                  # Sobol initial samples
│   ├── sobol_0/
│   │   ├── od.xml
│   │   └── sim_*.xml
│   └── sobol_{n_init-1}/
│
├── bo_iterations/                   # BO iteration simulations
│   ├── bo_iter_0/
│   │   ├── od.xml
│   │   └── sim_*.xml
│   └── bo_iter_{n_iter-1}/
│
├── results/                         # BO results and convergence
│   ├── convergence.csv              # Best restart convergence
│   ├── bo_results.pkl               # Best restart full results
│   ├── metadata.json                # Best restart metadata
│   ├── restart_1_seed-42/
│   │   ├── bo_results_restart_1.pkl
│   │   ├── convergence.csv
│   │   ├── edge_stats.csv
│   │   ├── metadata.json
│   │   └── convergence.png
│   └── restart_{n}_seed-{seed}/
│
└── saved_results/                   # Notebook format (optional)
    ├── {model}_bo_histories.json
    └── {model}_bo_histories.csv
```

---

## Notes

- **Consistent Seeds**: Both `optimization.py` and notebooks use the same seed calculation: `seed = 42 + (restart_idx - 1) * 1000`
- **Same Directory**: Notebooks and `optimization.py` save to the same directory structure, so results are interchangeable
- **Load Without Re-running**: Use `load_bo_results()` to load saved results and plot without re-running BO
- **Cleanup**: Set `CLEANUP_INTERMEDIATE_FILES=True` in config to remove intermediate simulation files after evaluation
