"""
ICM (Intrinsic Coregionalization Model) multi-output GP with multiple BO restarts.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from src.utils.seed import set_seed
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.link_flow_analysis import compute_squared_metric_per_edge


from MOGP.helpers_MOGP import initialize_icm_gp, LinearAggregationICM, fit_gpytorch_mll


def icm_model_multiple_restarts_multiple_BO_iterations(
    config,
    train_X_icm_init,
    train_Y_raw_init,
    gt_od_df,
    gt_edge_data,
    edge_ids,
    base_path,
    bounds,
    device,
    dtype,
    weights_uniform,
    routes_df,
):
    """
    Run ICM BO with multiple restarts; save convergence per restart and aggregated results.
    Returns icm_histories_raw, icm_histories_std, icm_seeds_used.
    """
    N_BO_ITERS = config["N_BO_ITERS"]
    N_RESTARTS = config["N_RESTART"]
    rank = config["rank"]
    d = bounds.shape[1]
    bounds_norm = torch.stack(
        [
            torch.zeros(d, dtype=dtype, device=device),
            torch.ones(d, dtype=dtype, device=device),
        ]
    )

    icm_histories_raw = []
    icm_histories_std = []
    icm_seeds_used = []

    for r in range(N_RESTARTS):
        print(f"      ICM RESTART {r+1}/{N_RESTARTS}")

        seed = 42 + r * 1000
        set_seed(seed)
        icm_seeds_used.append(seed)

        # Reset data
        train_X_icm = train_X_icm_init.clone()
        train_Y_icm_raw = train_Y_raw_init.clone()

        # Standardize per task (fixed stats per restart)
        y_mean = train_Y_icm_raw.mean(dim=0, keepdim=True)
        y_std = train_Y_icm_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_Y_icm = (train_Y_icm_raw - y_mean) / y_std

        # Initial best (raw + std)
        S_all_raw = (train_Y_icm_raw * weights_uniform).sum(dim=1)
        S_all_std = (train_Y_icm * weights_uniform).sum(dim=1)
        best_S_raw = S_all_raw.min().item()
        best_S_std = S_all_std.min().item()
        S_history_raw = [best_S_raw]
        S_history_std = [best_S_std]

        print(f"[INIT] Initial points: {train_X_icm.shape[0]}")
        print(f"[INIT] Best initial S (raw) = {best_S_raw:.6f}")

        for it in range(N_BO_ITERS):
            print(f"\n--- ICM BO ITER {it+1}/{N_BO_ITERS} ---")

            train_X_norm = normalize(train_X_icm, bounds=bounds)
            icm_model, mll = initialize_icm_gp(train_X_norm, train_Y_icm, rank=rank)
            icm_model = icm_model.to(device=device, dtype=dtype)
            icm_model.likelihood = icm_model.likelihood.to(device=device, dtype=dtype)
            mll = mll.to(device=device, dtype=dtype)
            agg_model = LinearAggregationICM(icm_model, weights_uniform).to(device=device, dtype=dtype)

            icm_model.train()
            mll.train()
            fit_gpytorch_mll(mll)
            icm_model.eval()
            agg_model.eval()

            EI = ExpectedImprovement(model=agg_model, best_f=best_S_std, maximize=False)
            x_next_norm, acq_val = optimize_acqf(
                EI,
                bounds=bounds_norm,
                q=1,
                num_restarts=config["NUM_RESTARTS"],
                raw_samples=config["RAW_SAMPLES"],
            )
            x_next_norm = x_next_norm.detach()
            x_next_real = unnormalize(x_next_norm, bounds).view(-1).cpu().numpy()

            print(f"[ACQ] EI = {acq_val.item():.6f}")
            print(f"[ACQ] x_next_real sum = {x_next_real.sum():.2f}")

            # RUN SUMO + COMPUTE PER-EDGE ERRORS
            sim_dir = f"{config['simulation_run_path']}/icm_bo_restart_{r}/iter_{it}"
            Path(sim_dir).mkdir(parents=True, exist_ok=True)
            new_od_xml = f"{sim_dir}/od.xml"
            prefix_output = f"{sim_dir}/sim"

            base_od = gt_od_df.copy()
            base_od["count"] = x_next_real
            base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})
            create_taz_xml(new_od_xml, base_od, config["od_duration_sec"], base_path)

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

            sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
            curr_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
                base_dir=base_path,
                sim_edge_file=sim_edge_out,
                prefix_output=prefix_output,
                SUMO_PATH=config["SUMO_PATH"],
            )
            e_next = compute_squared_metric_per_edge(
                df_true=gt_edge_data,
                df_simulated=curr_loop_stats,
                edge_ids=edge_ids,
            )

            e_next_tensor = torch.tensor(e_next, device=device, dtype=dtype)
            e_next_std = (e_next_tensor - y_mean.view(-1)) / y_std.view(-1)
            S_next_raw = (e_next_tensor * weights_uniform).sum().item()
            S_next_std = (e_next_std * weights_uniform).sum().item()

            print(f"[SIM] S_next_raw = {S_next_raw:.6f}, S_next_std = {S_next_std:.6f}")

            if S_next_raw < best_S_raw:
                improvement = best_S_raw - S_next_raw
                best_S_raw = S_next_raw
                print(f"[UPDATE] New BEST S (raw) = {best_S_raw:.6f} (Δ={improvement:.6f})")
            else:
                print(f"[NO IMPROVEMENT] Best S (raw) remains {best_S_raw:.6f}")
            if S_next_std < best_S_std:
                best_S_std = S_next_std

            S_history_raw.append(best_S_raw)
            S_history_std.append(best_S_std)

            train_X_icm = torch.cat(
                [train_X_icm, torch.tensor(x_next_real, device=device, dtype=dtype).view(1, -1)],
                dim=0,
            )
            train_Y_icm_raw = torch.cat([train_Y_icm_raw, e_next_tensor.view(1, -1)], dim=0)
            train_Y_icm = torch.cat([train_Y_icm, e_next_std.view(1, -1)], dim=0)
            print(f"[DATA] Training size = {train_X_icm.shape[0]}")

        icm_histories_raw.append(S_history_raw)
        icm_histories_std.append(S_history_std)
        print(f"\n[RESTART COMPLETE] Final best S = {best_S_raw:.6f}")

        # Save convergence for this restart immediately (so we have it if we stop early)
        results_dir = Path(config["simulation_run_path"]) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        restart_seed = icm_seeds_used[r]
        restart_dir = results_dir / f"restart_{r+1}_seed-{restart_seed}_rank{rank}"
        restart_dir.mkdir(parents=True, exist_ok=True)
        conv_df = pd.DataFrame({
            "iteration": range(len(S_history_raw)),
            "best_S_model": S_history_raw,
        })
        conv_df.to_csv(restart_dir / f"convergence_rank{rank}.csv", index=False)
        print(f"✓ Convergence saved: {restart_dir / f'convergence_rank{rank}.csv'}")

    # Mean across restarts (raw)
    min_len = min(len(h) for h in icm_histories_raw)
    S_history_raw_mean = np.mean([h[:min_len] for h in icm_histories_raw], axis=0).tolist()

    print("\nFinished ICM BO.")
    print(f"Final best S (raw, mean) = {S_history_raw_mean[-1]:.6f}")

    # Save aggregated results
    save_dir = Path(config["simulation_run_path"]) / "saved_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    icm_results = {
        "histories": icm_histories_raw,
        "histories_std": icm_histories_std,
        "mean_history": S_history_raw_mean,
        "num_restarts": len(icm_histories_raw),
        "iterations_per_restart": N_BO_ITERS,
        "config": {
            "NUM_RESTARTS": N_RESTARTS,
            "NITER": N_BO_ITERS,
            "rank": rank,
            "network_name": config.get("network_name", "quickstart"),
        },
    }
    with open(save_dir / f"icm_bo_histories_rank{rank}.json", "w") as f:
        json.dump(icm_results, f, indent=2)
    df_histories = pd.DataFrame({f"Restart_{i+1}": h for i, h in enumerate(icm_histories_raw)})
    df_histories["Mean"] = S_history_raw_mean[: len(df_histories)]
    df_histories.index.name = "Iteration"
    df_histories.to_csv(save_dir / f"icm_bo_histories_rank{rank}.csv")

    print(f"\n✓ Detailed format saved to: {save_dir}")
    print(f"  - {save_dir / f'icm_bo_histories_rank{rank}.json'}")
    print(f"  - {save_dir / f'icm_bo_histories_rank{rank}.csv'}")
    print(f"\n✓ Convergence saved to: {config['simulation_run_path']}/results/restart_X_seed-Y_rank{rank}/")

    return icm_histories_raw, icm_histories_std, icm_seeds_used
