# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd

import time
from pathlib import Path



# Local application imports
from src.simulation.sumo_runner import (
    create_od_tazrelation_xml,
    simulate_od,
    parse_loop_data_xml_to_pandas,
)
from src.utils.link_flow_analysis import (
    parse_link_flow_xml_to_pandas,
    compute_nrmse_all_links,
)




def run_single_od_evaluation(
    x,
    base_od,
    config,
    base_path,
    path_run_detail,
    path_run_simul,
    path_run_result,
    routes_df,
    routes_per_od,
    link_selection,
    eval_measure,
    sensor_measure_gt,
):
    """
    Run a simulation for a single OD (Origin-Destination) vector and record the results.

    Parameters
    ----------
    x : array-like
        OD demand values for the current sample.
    base_od : pd.DataFrame
        Base OD matrix DataFrame.
    config : dict
        Simulation and optimization configuration parameters.
    base_path : str
        Base directory for input/output files.
    path_run_detail : str
        Directory path to save runtime metadata.
    path_run_simul : str
        Directory path to save simulation outputs.
    path_run_result : str
        Directory path to save evaluation results.
    routes_df : pd.DataFrame
        Route information DataFrame.
    routes_per_od : str
        Type of routes to use for the simulation (single or multiple).
    link_selection : list
        List of links selected for evaluation.
    sensor_flow_gt : pd.DataFrame
        Ground truth sensor data for comparison.

    Returns
    -------
    pd.DataFrame
        DataFrame containing simulated link flow statistics for the current OD sample.
    """
    print("\n########### Start simulation and evaluation ###########")

    # Prepare file paths
    new_od_xml = f"{path_run_simul}/od.xml"
    prefix_output_simul = f"{path_run_simul}/result"

    # Prepare OD matrix
    curr_od = np.array(x)
    print(f"Total expected demand: {curr_od.sum():.1f}")

    base_od_copy = base_od.copy()
    base_od_copy["count"] = [round(elem, 1) for elem in curr_od]
    base_od_copy = base_od_copy.rename(columns={"fromTaz": "from", "toTaz": "to"})

    # Save OD as TAZ XML
    create_od_tazrelation_xml(
        od_df=base_od_copy,
        output_file=Path(new_od_xml),
        od_end_time_seconds=config["od_end_time"],
    )

    # Run SUMO simulation
    start_time = time.time()
    simulate_od(
        new_od_xml,
        prefix_output_simul,
        base_path,
        config["net_xml"],
        config["taz_xml"],
        config["additional_xml"],
        routes_df,
        routes_per_od,
        config["sim_end_time"],
        config["trips_xml_out_str"],
    )
    run_time = time.time() - start_time

    # Save simulation run time to a file
    run_time_hours, rem = divmod(run_time, 3600)
    run_time_minutes, run_time_seconds = divmod(rem, 60)
    run_time_str = f"simulation run time {int(run_time_hours)}h {int(run_time_minutes)}m {int(run_time_seconds)}s"
    run_time_file = Path(path_run_detail) / f"{run_time_str}.txt"
    with open(run_time_file, "w") as f:
        f.write(run_time_str)

    # Load simulation output and compute loss
    sim_link_out = f"{base_path}/{prefix_output_simul}_{config['link_data_out_str']}"
    curr_link_stats, _, _ = parse_link_flow_xml_to_pandas(
        base_path,
        sim_link_out,
        prefix_output_simul,
        config["sensor_start_time"],
        config["sensor_end_time"],
        link_list=link_selection,
    )
    curr_loss = compute_nrmse_all_links(eval_measure, sensor_measure_gt, curr_link_stats)
    print(f"Loss: {curr_loss:.4f}")

    # Merge ground truth and simulated flow data
    # Note: This format is compatible with visualization scripts
    if eval_measure == 'count':
        # Keep original column names for compatibility with fitGT plots
        sensor_measure_gt_temp = sensor_measure_gt[["link_id", "interval_nVehContrib"]].copy()
        sensor_measure_gt_temp = sensor_measure_gt_temp.rename(columns={"interval_nVehContrib": "flow_gt"})
        curr_link_stats_temp = curr_link_stats[["link_id", "interval_nVehContrib"]].copy()
        curr_link_stats_temp = curr_link_stats_temp.rename(columns={"interval_nVehContrib": "flow_simul"})
    elif eval_measure == 'speed':
        sensor_measure_gt_temp = sensor_measure_gt[["link_id", "interval_harmonicMeanSpeed"]].copy()
        sensor_measure_gt_temp = sensor_measure_gt_temp.rename(columns={"interval_harmonicMeanSpeed": "speed_gt"})
        curr_link_stats_temp = curr_link_stats[["link_id", "interval_harmonicMeanSpeed"]].copy()
        curr_link_stats_temp = curr_link_stats_temp.rename(columns={"interval_harmonicMeanSpeed": "speed_simul"})
    else:
        raise ValueError(f"Invalid eval_measure: {eval_measure}. Must be 'count' or 'speed'.")
    
    merged_data = pd.merge(sensor_measure_gt_temp, curr_link_stats_temp, on="link_id", how="inner")

    # Save to CSV (compatible format for analysis)
    output_csv_path = Path(path_run_result) / "link_measure_compare.csv"
    merged_data.to_csv(output_csv_path, index=False)
    print(f"Link measure comparison saved to {output_csv_path}")
    
    # Also save in format compatible with fitGT visualization (if epoch/batch info available)
    # Note: Convergence tracking (convergence.csv) is handled at the BO loop level, not here
    # This function only handles single evaluation results

    # Save NRMSE loss to a file
    nrmse_file = Path(path_run_result) / f"NRMSE_{curr_loss:.4f}.txt"
    with open(nrmse_file, "w") as f:
        f.write(f"NRMSE: {curr_loss:.4f}")

    # Clean up intermediate simulation files (optional)
    if config["eliminate_sumo_run_files"] == "True":
        trips_out = config["trips_xml_out_str"]
        try:
            os.remove(sim_link_out)
            os.remove(f"{base_path}/{prefix_output_simul}_{trips_out[:-4]}_beforeRteUpdates.xml")
            os.remove(f"{base_path}/{prefix_output_simul}_{trips_out}")
            # os.remove(f"{base_path}/{prefix_output_simul}_routes.vehroutes.xml")
        except FileNotFoundError:
            print("[Warning] Some intermediate files were not found during cleanup.")

    return curr_link_stats