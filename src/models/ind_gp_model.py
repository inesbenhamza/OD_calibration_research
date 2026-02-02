from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import json

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.utils.seed import set_seed
from src.models.gp_models import (
    initialize_independent_gp_models_with_modellist,
    make_linear_aggregation_model_from_error_gps,
)

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_X_ind_init = torch.tensor(
    np.array(train_X_list), dtype=dtype, device=device
)  # [N, d]
train_Y_ind_init = torch.tensor(
    np.array(train_E_list), dtype=dtype, device=device
)  # error per edge [N, L]

N, d = train_X_ind_init.shape
L = train_Y_ind_init.shape[1]
assert len(edge_ids) == L, f"edge_ids ({len(edge_ids)}) must match L ({L})"

# Aggregation weights: S(x) = sum_l w_l * e_l(x)
weights_t = torch.tensor(weights_np, dtype=dtype, device=device)  # uniform

# Bounds
max_demand = 2000.0
bounds = torch.stack(
    [
        torch.zeros(d, dtype=dtype, device=device),
        torch.full((d,), max_demand, dtype=dtype, device=device),
    ]
)
bounds_norm = torch.stack(
    [
        torch.zeros(d, dtype=dtype, device=device),
        torch.ones(d, dtype=dtype, device=device),
    ]
)

N_RESTARTS = config.get("NUM_RESTARTS", 1)
N_BO_ITERS = config.get("NITER", 10)

# Get kernel from config or default
kernel = config.get("kernel", "matern-2p5")

indgp_histories_raw = []

# BO loop
for r in range(N_RESTARTS):
    print(f"\n######### IND-GP BO RESTART {r+1}/{N_RESTARTS} #########")

    seed = 42 + r * 1000  # Same seed scheme as vanilla BO: 42, 1042, 2042, ...
    set_seed(seed)
    print(f"[SEED] Set seed to {seed}")

    train_X_ind = train_X_ind_init.clone()
    train_Y_ind = train_Y_ind_init.clone()  # [N, L]

    # Initial aggregated objective (RAW)
    S_init = (train_Y_ind * weights_t).sum(dim=1)
    best_S_raw = S_init.min().item()
    S_history_raw = [best_S_raw]
    S_obs_history_raw = S_init.detach().cpu().tolist()

    print(f"[INIT] Initial points: {train_X_ind.shape[0]}")
    print(f"[INIT] Best initial S (raw) = {best_S_raw:.6f}")

    # ============================================================
    # STOPPING CRITERIA CONFIGURATION
    # ============================================================
    EARLY_STOP_PATIENCE = config.get(
        "EARLY_STOP_PATIENCE", 0
    )  # 0 = disabled, N = stop after N iterations without improvement
    EARLY_STOP_DELTA = config.get(
        "EARLY_STOP_DELTA", 1e-6
    )  # Minimum improvement threshold
    EARLY_STOP_MIN_ACQ = config.get(
        "EARLY_STOP_MIN_ACQ", 0.0
    )  # Minimum acquisition value (0 = disabled)
    no_improve_steps = 0
    best_S_for_stop = best_S_raw  # Track best for stopping criteria

    # Print early stopping configuration if enabled
    if EARLY_STOP_PATIENCE > 0:
        print(f"\n{'='*60}")
        print(f"EARLY STOPPING ENABLED")
        print(f"{'='*60}")
        print(f"  Patience: {EARLY_STOP_PATIENCE} iterations")
        print(f"  Minimum improvement: {EARLY_STOP_DELTA:.6f}")
        print(f"  Initial best S: {best_S_for_stop:.6f}")
        print(f"{'='*60}\n")

    for it in range(N_BO_ITERS):
        print(f"\n######### IND-GP ITERATION {it+1}/{N_BO_ITERS} #########")

        # Normalize inputs
        train_X_norm = normalize(train_X_ind, bounds=bounds)

        # Fit independent per-edge GPs
        model_list_gp_errors, mlls = initialize_independent_gp_models_with_modellist(
            train_X_norm,
            train_Y_ind,
            kernel=kernel,  # Use kernel from config
        )  # here we get one gp per edge

        # Fit each edge's GP (mlls is a list, one MLL per edge)
        for i, mll in enumerate(mlls):
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"[WARNING] Edge {edge_ids[i]} fit warning: {str(e)[:80]}")

        # Build aggregation model S(x)
        agg_model = make_linear_aggregation_model_from_error_gps(
            model_list_gp_errors,
            weights=weights_t,
        )  # here we compute the posterior error prediction for each edge at a new point x
        # Aggregates them using the weights to get the final aggregated error prediction
        # It computes both the mean and variance for each edge, then aggregates both
        # this is needed as EI expect a scalar output
        agg_model.eval()

        EI = ExpectedImprovement(
            model=agg_model,
            best_f=best_S_raw,
            maximize=False,
        )
        # raw space bc When you define EI = ExpectedImprovement(model, best_f), the acquisition function is a callable object.
        # When you call it (e.g., EI(test_X)), it internally executes:
        # model.posterior(X): This generates a Posterior object representing the model's belief at those points.
        # EI.forward(X): This computes the Expected Improvement at the points X.
        # The acquisition function is a callable object that takes a tensor of test points X and returns the Expected Improvement at those points.

        x_next_norm, acq_val = optimize_acqf(
            acq_function=EI,
            bounds=bounds_norm,
            q=1,
            num_restarts=config.get("NUM_RESTARTS", 5),
            raw_samples=config.get("RAW_SAMPLES", 32),
        )

        x_next_norm = x_next_norm.detach()
        x_next_real = (
            unnormalize(x_next_norm, bounds=bounds).squeeze(0).cpu().numpy()
        )

        restart_seed = seed
        sim_dir = (
            f"{config['simulation_run_path']}/"
            f"indgp_restart_{r}_seed-{restart_seed}/iter_{it}"
        )
        Path(sim_dir).mkdir(parents=True, exist_ok=True)

        prefix_output = f"{sim_dir}/sim"
        new_od_xml = f"{sim_dir}/od.xml"

        base_od = gt_od_df.copy()
        base_od["count"] = [round(v, 1) for v in x_next_real]
        base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})

        create_taz_xml(
            new_od_xml,
            base_od,
            config["od_duration_sec"],
            base_path,
        )

        t0 = time.time()
        simulate_od(
            new_od_xml,
            prefix_output,
            base_path,
            config["net_xml"],
            config["taz2edge_xml"],
            config["additional_xml"],
            routes_df,
            config["sim_end_time"],
            config["TRIPS2ODS_OUT_STR"],
        )
        sim_time = time.time() - t0
        print(f"[RUNTIME] Simulation: {sim_time:.2f}s")

        # ----------------------------------------------------
        # Parse results
        # ----------------------------------------------------
        sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
        curr_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
            base_dir=base_path,
            sim_edge_file=sim_edge_out,
            prefix_output=prefix_output,
            SUMO_PATH=config["SUMO_PATH"],
        )

        per_edge_errors = compute_squared_metric_per_edge(
            df_true=gt_edge_data,
            df_simulated=curr_loop_stats,
            edge_ids=edge_ids,
        )

        curr_loss = float((per_edge_errors * weights_np).sum())
        if np.isnan(curr_loss):
            curr_loss = float("inf")

        print(f"[SIM] S(x_next) = {curr_loss:.6f}")

        # ----------------------------------------------------
        # Update tracking
        # ----------------------------------------------------
        S_obs_history_raw.append(curr_loss)

        if curr_loss < best_S_raw:
            improvement = best_S_raw - curr_loss
            print(f"[UPDATE] New BEST S = {curr_loss:.6f} (Δ={improvement:.6f})")
            best_S_raw = curr_loss
        else:
            print(f"[NO IMPROVEMENT] Best S remains {best_S_raw:.6f}")

        S_history_raw.append(best_S_raw)

        # Save results for this iteration
        # Use sim_dir as the output directory
        path_run_result = Path(sim_dir)

        # Create merged comparison data (if you want to save the comparison)
        merged_data = pd.merge(
            gt_edge_data[["edge_id", "interval_nVehContrib"]],
            curr_loop_stats[["edge_id", "interval_nVehContrib"]],
            on="edge_id",
            suffixes=("_gt", "_sim"),
        )

        output_csv_path = path_run_result / "link_measure_compare.csv"
        merged_data.to_csv(output_csv_path, index=False)
        print(f"Link measure comparison saved to {output_csv_path}")

        # Save NRMSE loss to a file
        nrmse_file = path_run_result / f"NRMSE_{curr_loss:.6f}.txt"  # loss value at each iteration
        with open(nrmse_file, "w") as f:
            f.write(f"NRMSE: {curr_loss:.6f}")

        # Clean up intermediate simulation files (optional)
        if config.get("CLEANUP_INTERMEDIATE_FILES", False):
            try:
                # Remove simulation output files
                if Path(sim_edge_out).exists():
                    Path(sim_edge_out).unlink()
                # Add other cleanup as needed
            except Exception as e:
                print(f"[Warning] Cleanup error: {e}")

        # ============================================================
        # CHECK STOPPING CRITERIA (INSIDE LOOP - properly indented)
        # ============================================================
        if EARLY_STOP_PATIENCE > 0:
            # Calculate improvement from the best metric tracked so far
            improvement = best_S_for_stop - best_S_raw

            # Check if we have significant improvement
            if improvement >= EARLY_STOP_DELTA:
                # Significant improvement found - reset counter and update best
                best_S_for_stop = best_S_raw
                no_improve_steps = 0
                print(
                    f"  [EARLY-STOP] Improvement of {improvement:.6f} ≥ {EARLY_STOP_DELTA:.6f} "
                    f"- counter reset"
                )
            else:
                # No significant improvement
                no_improve_steps += 1
                print(
                    f"  [EARLY-STOP] Counter: {no_improve_steps}/{EARLY_STOP_PATIENCE} "
                    f"(improvement {improvement:.6f} < {EARLY_STOP_DELTA:.6f})"
                )
                if no_improve_steps >= EARLY_STOP_PATIENCE:
                    print(
                        f"\n{'='*60}\n"
                        f"EARLY STOPPING TRIGGERED\n"
                        f"{'='*60}\n"
                        f"No improvement ≥ {EARLY_STOP_DELTA:.6f} "
                        f"for {EARLY_STOP_PATIENCE} consecutive iterations.\n"
                        f"  Best S achieved: {best_S_for_stop:.6f}\n"
                        f"  Final S: {best_S_raw:.6f}\n"
                        f"  Stopped at iteration {it+1}/{N_BO_ITERS}\n"
                        f"{'='*60}\n"
                    )
                    break

            # Always update best_S_for_stop if current is better (for tracking)
            if best_S_raw < best_S_for_stop:
                best_S_for_stop = best_S_raw

        # Check acquisition value threshold (optional)
        if EARLY_STOP_MIN_ACQ > 0 and acq_val.item() < EARLY_STOP_MIN_ACQ:
            print(
                f"\n{'='*60}\n"
                f"EARLY STOPPING TRIGGERED (Acquisition Threshold)\n"
                f"{'='*60}\n"
                f"Acquisition value {acq_val.item():.6f} < threshold {EARLY_STOP_MIN_ACQ:.6f}\n"
                f"  Best S: {best_S_raw:.6f}\n"
                f"  Stopped at iteration {it+1}/{N_BO_ITERS}\n"
                f"{'='*60}\n"
            )
            break

        # ----------------------------------------------------
        # Update dataset
        # ----------------------------------------------------
        train_X_ind = torch.cat(
            [
                train_X_ind,
                torch.tensor(x_next_real, device=device, dtype=dtype).view(1, -1),
            ],
            dim=0,
        )
        train_Y_ind = torch.cat(
            [
                train_Y_ind,
                torch.tensor(per_edge_errors, device=device, dtype=dtype).view(1, -1),
            ],
            dim=0,
        )

        print(f"[DATA] Training size = {train_X_ind.shape[0]}")

    indgp_histories_raw.append(S_history_raw)
    print(f"[RESTART COMPLETE] Final best S = {best_S_raw:.6f}")

# ============================================================
# Aggregate across restarts
# ============================================================
if len(indgp_histories_raw) > 1:
    min_len = min(len(h) for h in indgp_histories_raw)
    S_history_raw_mean_ind = np.mean(
        [h[:min_len] for h in indgp_histories_raw],
        axis=0,
    ).tolist()
else:
    S_history_raw_mean_ind = indgp_histories_raw[0]

print("\n============================================================")
print("IND-GP BO COMPLETE")
print("============================================================")
print(
    f"Final best S (mean across restarts) = {S_history_raw_mean_ind[-1]:.6f}"
)
print(f"Restarts: {N_RESTARTS}, Iterations per restart: {N_BO_ITERS}")

# ============================================================
# SAVE IN BOTH FORMATS: Detailed + Visualization-compatible
# ============================================================
# 1. Save detailed format (your current format)
save_dir = Path(config["simulation_run_path"]) / "saved_results"
save_dir.mkdir(parents=True, exist_ok=True)

# Save histories (best loss at each iteration for each restart)
indgp_results = {
    "histories": indgp_histories_raw,  # List of lists: [restart][iteration]
    "mean_history": S_history_raw_mean_ind,
    "num_restarts": len(indgp_histories_raw),
    "iterations_per_restart": N_BO_ITERS,
    "kernel": kernel,  # Include kernel information
    "config": {
        "NUM_RESTARTS": N_RESTARTS,
        "NITER": N_BO_ITERS,
        "kernel": kernel,
        "network_name": config.get("network_name", "quickstart"),
    },
}

# Save as JSON
with open(save_dir / "indgp_bo_histories.json", "w") as f:
    json.dump(indgp_results, f, indent=2)

# Also save as CSV for easy viewing
df_histories = pd.DataFrame(
    {f"Restart_{r+1}": history for r, history in enumerate(indgp_histories_raw)}
)
df_histories["Mean"] = S_history_raw_mean_ind[: len(df_histories)]
df_histories.index.name = "Iteration"
df_histories.to_csv(save_dir / "indgp_bo_histories.csv")

print(f"\n✓ Detailed format saved to: {save_dir}")
print(f"  - Histories JSON: {save_dir / 'indgp_bo_histories.json'}")
print(f"  - Histories CSV: {save_dir / 'indgp_bo_histories.csv'}")

# 2. ALSO save in format expected by results_visualization.py
#    This creates results/convergence.csv files (one per restart)
results_dir = Path(config["simulation_run_path"]) / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Save convergence.csv for each restart (format expected by results_visualization.py)
for r, history in enumerate(indgp_histories_raw):
    restart_seed = 42 + r * 1000
    restart_dir = results_dir / f"restart_{r+1}_seed-{restart_seed}"
    restart_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame in the format expected by visualization script
    conv_df = pd.DataFrame(
        {
            "iteration": range(len(history)),
            "best_S_model": history,  # Your best-so-far values
        }
    )

    conv_df.to_csv(restart_dir / "convergence.csv", index=False)
    print(f"✓ Visualization format saved: {restart_dir / 'convergence.csv'}")

print(f"\n✓ Visualization format saved to: {results_dir}")
print(f"  - Kernel: {kernel}")
print(f"  - Format: results/restart_X_seed-Y/convergence.csv")
