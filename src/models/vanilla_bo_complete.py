from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import json

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.utils.seed import set_seed
from src.models.gp_models import initialize_vanillabo_model


dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N, d = train_X.shape
assert train_Y.shape == (N, 1), f"train_Y must be [N, 1], got {train_Y.shape}"


# Bounds for the input 
max_demand = 2000.0

bounds = torch.stack([
    torch.zeros(d, dtype=dtype, device=device),
    torch.full((d,), max_demand, dtype=dtype, device=device),
])

# bound for the acquisition function 
bounds_norm = torch.stack([
    torch.zeros(d, dtype=dtype, device=device),
    torch.ones(d, dtype=dtype, device=device),
])


# Initial data 
train_X_van_init = train_X.clone()  # [N, d]
train_Y_van_init = train_Y.clone()  # [N, 1] raw aggregated loss


N_RESTARTS = config.get("NUM_RESTARTS", 1)
N_BO_ITERS = config.get("NITER", 30)

# Get kernel from config or default
kernel = config.get("kernel", "matern-2p5")

vanilla_histories_raw = []

# BO loop 
for r in range(N_RESTARTS):
    print(f"\n######### VANILLA BO RESTART {r+1}/{N_RESTARTS} #########")

    seed = 42 + r * 1000  
    set_seed(seed)
    print(f"[SEED] Set seed to {seed}")

    train_X_van = train_X_van_init.clone()
    train_Y_van = train_Y_van_init.clone()

    # Best observed value in non standardized space
    best_S_raw = train_Y_van.min().item()
    S_history_raw = [best_S_raw]
    S_obs_history_raw = train_Y_van.squeeze().detach().cpu().tolist()

    print(f"[INIT] Initial points: {train_X_van.shape[0]}")
    print(f"[INIT] Best initial S (raw) = {best_S_raw:.6f}")

    # ============================================================
    # STOPPING CRITERIA CONFIGURATION
    # ============================================================
    EARLY_STOP_PATIENCE = config.get("EARLY_STOP_PATIENCE", 0)  # 0 = disabled, N = stop after N iterations without improvement
    EARLY_STOP_DELTA = config.get("EARLY_STOP_DELTA", 1e-6)  # Minimum improvement threshold
    EARLY_STOP_MIN_ACQ = config.get("EARLY_STOP_MIN_ACQ", 0.0)  # Minimum acquisition value (0 = disabled)
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
        print(f"\n######### VANILLA BO ITERATION {it+1}/{N_BO_ITERS} #########")

        # Normalize inputs
        train_X_norm = normalize(train_X_van, bounds=bounds)

        # Fit GP
        model = initialize_vanillabo_model(
            train_X_norm,
            train_Y_van,
            kernel=kernel,  # Use kernel from config
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        best_f_for_ei = train_Y_van.view(-1).min()

        print(
            f"[MODEL] best_S_raw = {best_S_raw:.6f}, "
            f"best_f_for_ei = {best_f_for_ei.item():.6f}"
        )

        acquisition_function = ExpectedImprovement(
            model=model,
            best_f=best_f_for_ei,
            maximize=False,
        )

        # Optimizing acquisition EI
        x_next_norm, acq_val = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds_norm,
            q=1,
            num_restarts=32,
            raw_samples=128,
        )

        x_next_norm = x_next_norm.detach()
        x_next_real = (
            unnormalize(x_next_norm, bounds)
            .view(-1)
            .cpu()
            .numpy()
        )

        print(f"[ACQ] EI value = {acq_val.item():.6f}")
        print(f"[ACQ] x_next (real) sum = {x_next_real.sum():.2f}")

        restart_seed = seed
        sim_dir = (
            f"{config['simulation_run_path']}/"
            f"vanilla_bo_restart_{r}_seed-{restart_seed}/iter_{it}"
        )
        Path(sim_dir).mkdir(parents=True, exist_ok=True)

        new_od_xml = f"{sim_dir}/od.xml"
        prefix_output = f"{sim_dir}/sim"

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
        print(f"[RUNTIME] Simulation took {sim_time:.2f}s")

        # Parse & compute loss
        sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
        curr_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
            base_dir=base_path,
            sim_edge_file=sim_edge_out,
            prefix_output=prefix_output,
            SUMO_PATH=config["SUMO_PATH"],
        )

        curr_loss = compute_squared_metric_all_edge(
            df_true=gt_edge_data,
            df_simulated=curr_loop_stats,
            edge_ids=edge_ids,
        )

        print(f"[SIM] S(x_next) = {curr_loss:.6f}")

        # Update history with new point
        S_obs_history_raw.append(curr_loss)

        if curr_loss < best_S_raw:
            improvement = best_S_raw - curr_loss
            print(f"[UPDATE] New BEST S = {curr_loss:.6f} (Δ={improvement:.6f})")
            best_S_raw = curr_loss
        else:
            print(f"[NO IMPROVEMENT] Best S remains {best_S_raw:.6f}")

        S_history_raw.append(best_S_raw)  # tracks the best value seen so far, not the observed value at each iteration.

        # Update dataset
        train_X_van = torch.cat(
            [train_X_van, torch.tensor(x_next_real, device=device, dtype=dtype).view(1, -1)],
            dim=0,
        )
        train_Y_van = torch.cat(
            [train_Y_van, torch.tensor([[curr_loss]], device=device, dtype=dtype)],
            dim=0,
        )

        print(f"[DATA] Training size = {train_X_van.shape[0]}")  # at the end of the loop, the training set has been updated with the new point.

        # Save results for this iteration
        # Use sim_dir as the output directory
        path_run_result = Path(sim_dir) 

        # Create merged comparison data (if you want to save the comparison)
        merged_data = pd.merge(
            gt_edge_data[['edge_id', 'interval_nVehContrib']],
            curr_loop_stats[['edge_id', 'interval_nVehContrib']],
            on='edge_id',
            suffixes=('_gt', '_sim')
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

    vanilla_histories_raw.append(S_history_raw)  # list of lists, each inner list contains the best loss values for each iteration across all restarts.
    print(f"\n[RESTART COMPLETE] Final best S = {best_S_raw:.6f}")


# Aggregate across restarts
if len(vanilla_histories_raw) > 1:
    min_len = min(len(h) for h in vanilla_histories_raw)
    S_history_raw_mean = np.mean(
        [h[:min_len] for h in vanilla_histories_raw],
        axis=0,
    ).tolist()
else:
    S_history_raw_mean = vanilla_histories_raw[0]


print("###VANILLA BO COMPLETE###")
print(f"Final best S (mean across restarts) = {S_history_raw_mean[-1]:.6f}")
print(f"Restarts: {N_RESTARTS}, Iterations per restart: {N_BO_ITERS}")


# ============================================================
# SAVE IN BOTH FORMATS: Detailed + Visualization-compatible
# ============================================================
# 1. Save detailed format (your current format)
save_dir = Path(config['simulation_run_path']) / 'saved_results'
save_dir.mkdir(parents=True, exist_ok=True)

# Save histories (best loss at each iteration for each restart)
vanilla_results = {
    'histories': vanilla_histories_raw,  # List of lists: [restart][iteration]
    'mean_history': S_history_raw_mean,
    'num_restarts': len(vanilla_histories_raw),
    'iterations_per_restart': N_BO_ITERS,
    'kernel': kernel,  # Include kernel information
    'config': {
        'NUM_RESTARTS': N_RESTARTS,
        'NITER': N_BO_ITERS,
        'kernel': kernel,
        'network_name': config.get('network_name', 'quickstart'),
    }
}

# Save as JSON
with open(save_dir / 'vanilla_bo_histories.json', 'w') as f:
    json.dump(vanilla_results, f, indent=2)

# Also save as CSV for easy viewing
df_histories = pd.DataFrame({
    f'Restart_{r+1}': history for r, history in enumerate(vanilla_histories_raw)
})
df_histories['Mean'] = S_history_raw_mean[:len(df_histories)]
df_histories.index.name = 'Iteration'
df_histories.to_csv(save_dir / 'vanilla_bo_histories.csv')

print(f"\n✓ Detailed format saved to: {save_dir}")
print(f"  - Histories JSON: {save_dir / 'vanilla_bo_histories.json'}")
print(f"  - Histories CSV: {save_dir / 'vanilla_bo_histories.csv'}")


# 2. ALSO save in format expected by results_visualization.py
#    This creates results/convergence.csv files (one per restart)
results_dir = Path(config['simulation_run_path']) / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

# Save convergence.csv for each restart (format expected by results_visualization.py)
for r, history in enumerate(vanilla_histories_raw):
    restart_seed = 42 + r * 1000
    restart_dir = results_dir / f"restart_{r+1}_seed-{restart_seed}"
    restart_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame in the format expected by visualization script
    conv_df = pd.DataFrame({
        "iteration": range(len(history)),
        "best_S_model": history,  # Your best-so-far values
        "best_S_common": history,  # Use same values if you don't have separate common metric
    })
    
    conv_df.to_csv(restart_dir / "convergence.csv", index=False)
    print(f"✓ Visualization format saved: {restart_dir / 'convergence.csv'}")

print(f"\n✓ Visualization format saved to: {results_dir}")
print(f"  - Kernel: {kernel}")
print(f"  - Format: results/restart_X_seed-Y/convergence.csv")
