# sumo environement  installation 

import os
from pathlib import Path 
import sys

# Add project root to Python path BEFORE other imports
# This allows imports like "from src.simulation..." to work
project_root = Path(__file__).resolve().parent.parent # Path(__file__).resolve()give  absolute path of the file being executed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch 
import pandas as pd 
import numpy as np
import pprint 

from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize


from helpers_od_calibration_BN import (BayesianOptimizationLoop, run_multiple_bo_restarts)
 

# Local imports
from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_od_xml, create_taz_xml, simulate_od
from src.simulation.evaluation import (
    parse_loop_data_xml_to_pandas,
)


from src.utils.seed  import set_seed

from src.utils.link_flow_analysis import (
    compute_squared_metric_all_edge,
    compute_squared_metric_per_edge,
    compute_cubic_metric_per_edge,
)
   



sumo_home = os.environ.get("SUMO_HOME")


##############################################################
#load sumo environment
##############################################################


# macOS-specific path 
if not sumo_home:
    macos_path = '/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/share/sumo'
    if os.path.exists(macos_path):
        sumo_home = macos_path
    else:
        # Try other default paths
        default_sumo_paths = [
            "/opt/sumo-1.12/share/sumo",  # Linux
            "C:/Program Files (x86)/Eclipse/Sumo",  # Windows
        ]
        sumo_home = next((p for p in default_sumo_paths if os.path.exists(p)), None)

# If still not found, exit with error
if not sumo_home:
    sys.exit("SUMO_HOME is not set and no default path exists.")

# Set the environment variable
os.environ['SUMO_HOME'] = sumo_home
os.environ['LIBSUMO_AS_TRACI'] = '1'  # Optional: for a huge performance boost (~8x) with Libsumo (No GUI)
SUMO_HOME = Path(os.environ['SUMO_HOME'])


# adding sumo tools to python path 
tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
if os.path.exists(tools_path):
    sys.path.append(tools_path)
else:
    sys.exit(f"Cannot find SUMO tools at {tools_path}")


# Add the SUMO *bin* directory to PATH
SUMO_ROOT = Path(os.environ["SUMO_HOME"]).parents[1]
SUMO_BIN  = str(SUMO_ROOT / "bin")
LIB_SUMO_PATH = Path(os.environ["SUMO_HOME"]).parents[2] / "lib" / "libsumo.dylib"
os.environ["PATH"] = SUMO_BIN + os.pathsep + os.environ.get("PATH", "")


# project basepath 
# (project_root already set above for sys.path)
base_path = str(project_root)

#/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn/src/optimization.py
#/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn

if ' ' in base_path:
    raise ValueError("base_path should not contain any spaces.")

os.chdir(base_path)



################################################ 


def main():
   # definin gcommand line arguments to set default and alllowed choices 

    parser = argparse.ArgumentParser(description="OD calibration")
    parser.add_argument("--network_name", type=str, default="1ramp", choices=["2corridor", "3junction", "4smallRegion", "5fullRegion", "quickstart"])
    parser.add_argument("--model_name", type=str, default="vanillabo", choices=["vanillabo", "independent_gp", "mogp"])
    parser.add_argument("--kernel", type=str, default="matern-2p5", choices=["matern-1p5", "matern-2p5", "rbf"])
    parser.add_argument("--date", type=int, default=221014, help="Date for simulation")
    parser.add_argument("--hour", type=str, default="08-09", choices=["06-07", "08-09", "17-18"], help="Time for simulation")
    parser.add_argument("--eval_measure", type=str, default="count", choices=["count", "speed"], help="Evaluation measurements")
    parser.add_argument("--routes_per_od", type=str, default="single", choices=["single", "multiple"], help="Type of routes to use for the simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cpu_max", type=int, default=6, help="Maximum number of CPU cores for parallel processing")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"], help="Device for computation")
    parser.add_argument("--n_iterations", type=int, default=30, help="Number of BO iterations")
    parser.add_argument("--n_restarts", type=int, default=5, help="Number of BO restarts")
    parser.add_argument("--error_metric", type=str, default="squared", choices=["squared", "cubic"], help="Error metric type")
    parser.add_argument("--weights", type=str, default="uniform", choices=["uniform", "flow_proportional"], help="Weighting scheme")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Early stopping patience (0 disables)")
    parser.add_argument("--early_stop_delta", type=float, default=0.0, help="Minimum improvement to reset patience")
    args = parser.parse_args() #Parses the command-line arguments
    


    


    #print(args)



 #Extracts values from the parsed arguments into local variables
    seed = args.seed
    set_seed(seed) # using function in misc.py
    date = args.date
    hour = args.hour
    model_name = args.model_name
    kernel = args.kernel
    network_name = args.network_name
    eval_measure = args.eval_measure
    routes_per_od = args.routes_per_od
    cpu_max = args.cpu_max
    n_iterations = args.n_iterations
    n_restarts = args.n_restarts

# exemple usage : python src/optimization.py --network_name quickstart --model_name vanillabo --kernel matern-2p5 --n_iterations 50


##############################################################
#loading config
##############################################################
  
    # network

    if network_name == "quickstart":
        sim_setup_filename = "sim_setup.json"
    else:
        sim_setup_filename = f"sim_setup_network_{network_name}.json"
    

    config = load_kwargs_config(
        base_path,
        model_name=model_name,
        sim_setup_filename=sim_setup_filename,
        kernel=kernel # use the kernel specified in the command line arguments, that we trasnformed into a local variable
    )

 
 

    Path(config["simulation_run_path"]).mkdir(parents=True, exist_ok=True)
    pprint.pprint(dict(config))


    #input data : gt od and routes

    print(f"Reading: {config['file_gt_od']}") # path to the gt od xml file
    gt_od_df = od_xml_to_df(config["file_gt_od"])
    

    dim_od = gt_od_df.shape[0]
    print(f"Number of OD pairs: {dim_od}")

    print(f"Reading: {config['fixed_routes']}")


    routes_csv = config["fixed_routes"]
    routes_df = pd.read_csv(routes_csv, index_col=0)

    # Set up GT simulation


    simulation_gt_run_path = f"{config['simulation_run_path']}/ground_truth"
    #output/quickstart_vanillabo_matern-2p5/ground_truth

    prefix_output_gt = f"{simulation_gt_run_path}/sim"
    #output/quickstart_vanillabo_matern-2p5/ground_truth/sim

    sim_edge_out_gt = f"{prefix_output_gt}_{config['EDGE_OUT_STR']}"
    #output/quickstart_vanillabo_matern-2p5/ground_truth/sim_edge_data.xml

    new_od_xml = f"{simulation_gt_run_path}/od.xml"
    #output/quickstart_vanillabo_matern-2p5/ground_truth/od.xml
    

    Path(simulation_gt_run_path).mkdir(parents=True, exist_ok=True)

    base_od = gt_od_df.copy()
    curr_od = base_od['count'].astype(float).to_numpy()
    base_od['count'] = curr_od
    base_od = base_od.rename(columns={'fromTaz':'from', 'toTaz':'to'})

    create_taz_xml(new_od_xml, base_od, config["od_duration_sec"], base_path)

    # Load ground-truth sensor measurements
    # For quickstart network, use GT simulation output instead of sensor data files
    true_sensor_file_name = f"gt_link_data_{network_name}_{date}_{hour}.csv"
    sensor_data_path = base_path + f"/sensor_data/{date}/" + true_sensor_file_name
    
    if network_name == "quickstart" or not Path(sensor_data_path).exists():
        # For quickstart or when sensor data doesn't exist, use GT simulation output
        print(f"Using ground truth simulation output (no sensor data file found)")
        # Run GT simulation first, then parse its output
        simulate_od(
            str(new_od_xml),
            prefix_output_gt,
            base_path,
            str(config["net_xml"]),
            str(config["taz2edge_xml"]),
            str(config["additional_xml"]),
            routes_df,
            str(config["sim_end_time"]),
            config["TRIPS2ODS_OUT_STR"],
        )
        # Parse GT simulation output
        gt_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
            base_path, sim_edge_out_gt, prefix_output_gt, str(config["SUMO_PATH"])
        )
        gt_edge_data = gt_loop_stats
        # Extract edge IDs from the simulation output
        if "edge_id" in gt_loop_stats.columns:
            edge_ids = gt_loop_stats["edge_id"].tolist()
        else:
            # Fallback: use all edges from network
            try:
                import sumolib
                net = sumolib.net.readNet(str(config["net_xml"]))
                edge_ids = [edge.getID() for edge in net.getEdges()]
            except ImportError:
                # If sumolib not available, extract from simulation output columns
                edge_ids = list(gt_loop_stats.columns) if hasattr(gt_loop_stats, 'columns') else []
    else:
        
        sensor_measure_gt = pd.read_csv(sensor_data_path)
        # Extract the list of links where sensors are located
        link_selection = sensor_measure_gt["link_id"].tolist()
        print(f"Number of sensors: {len(link_selection)}")
        gt_edge_data = sensor_measure_gt
        edge_ids = link_selection
    
    gt_od_vals = gt_od_df["count"].values


    device = torch.device(args.device)
    dtype = torch.double
    print(f"Using device: {device}")

    # Compute bounds from config (or use defaults)
    od_bound_start = config.get("od_bound_start", 0.0)
    od_bound_end = config.get("od_bound_end", 2000.0)
    bounds = torch.tensor(
        [[od_bound_start] * dim_od, [od_bound_end] * dim_od],
        device=device,
        dtype=dtype,
    )

    n_init_search = config["n_init_search"]


    if model_name == "vanillabo":
        error_metric_func = compute_squared_metric_all_edge
    elif model_name in ["independent_gp", "mogp"]:
        if args.error_metric == "cubic":
            error_metric_func = compute_cubic_metric_per_edge
        elif args.error_metric == "squared":
            error_metric_func = compute_squared_metric_per_edge
        else:
            print(f"Error metric {args.error_metric} not defined. Using squared error.")
            error_metric_func = compute_squared_metric_per_edge
    else:
        print(f"Model name '{model_name}' not recognized. Using default squared metric for vanillabo.")
        error_metric_func = compute_squared_metric_all_edge

    use_flow_proportional_weights = (args.error_metric == "cubic")
    print(
        "\n[DEBUG] BO configuration:"
        f" model={model_name}, error_metric={args.error_metric},"
        f" error_func={error_metric_func.__name__},"
        f" use_flow_proportional_weights={use_flow_proportional_weights}"
    )

    # paths for simulation run
 
    sim_run_path = Path(config["simulation_run_path"])
    sim_run_path.mkdir(parents=True, exist_ok=True)

    path_init = sim_run_path / "initial_search"
    path_init.mkdir(exist_ok=True)

    path_opt = sim_run_path / "bo_iterations"
    path_opt.mkdir(exist_ok=True)

    path_result = sim_run_path / "results"
    path_result.mkdir(exist_ok=True)

    print(f"\nOutput paths:")
    print(f"  Initial search: {path_init}")
    print(f"  BO iterations: {path_opt}")
    print(f"  Results: {path_result}")

    # Simple path for initial search simulations (replaces path_init_simul)
    path_init_simul = path_init


    # Run initial search

    print("RUNNING INITIAL SEARCH")
    
    # Generate  samples (normalized [0, 1])
    sobol = SobolEngine(dimension=dim_od, scramble=True, seed=seed)
    train_X_init_norm = sobol.draw(n_init_search).to(dtype=dtype, device=device)

    print(f"Generated {n_init_search} initial Sobol samples")
    print(f"  Shape: {train_X_init_norm.shape}")

    # Evaluate initial samples
    train_Y_errors_init = []
    for i in range(n_init_search):
        print(f"\nEvaluating initial sample {i+1}/{n_init_search}")

        # Unnormalize OD values
        X_real = unnormalize(train_X_init_norm[i:i+1], bounds).squeeze()
        curr_od = X_real.cpu().numpy()

        # Create OD XML
        sim_dir = path_init_simul / f"sobol_{i}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        od_xml = sim_dir / "od.xml"

        od_pairs = (
            routes_df[["fromTaz", "toTaz"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        base_od = od_pairs.copy()
        base_od["count"] = curr_od

        create_od_xml(
            curr_od,
            base_od,
            "od.xml",  # Just the filename, not full path
            config["od_duration_sec"],
            str(sim_dir),
        )

        # Run simulation
        prefix_output = str(sim_dir / "sim")
        simulate_od(
            str(od_xml),
            prefix_output,
            base_path,
            str(config["net_xml"]),
            str(config["taz2edge_xml"]),
            str(config["additional_xml"]),
            routes_df,
            str(config["sim_end_time"]),
            config["TRIPS2ODS_OUT_STR"],
        )

        # Parse results
        sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
        curr_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
            base_path, sim_edge_out, prefix_output, str(config["SUMO_PATH"])
        )

        # Compute per-edge errors (model-specific, for GP training)
        errors = error_metric_func(
        gt_edge_data,
         curr_loop_stats,edge_ids,)

        # Vanilla BO → aggregated scalar
        if model_name == "vanillabo":
            train_Y_errors_init.append(float(errors))
        # Independent GP → per-edge vector
        else:
            train_Y_errors_init.append(errors)

    # Convert to tensor (after all initial samples are evaluated)
    train_Y_errors_init = torch.tensor(
        np.array(train_Y_errors_init),
        dtype=dtype,
        device=device,
    )

    print(f"\nInitial search complete. Shape: {train_Y_errors_init.shape}")


    if model_name is not None :
        print("RUNNING OPTIMIZATION LOOP")
    


        #run multiple bo restarts 
        aggregated_results = run_multiple_bo_restarts(
            config=config,
            gt_edge_data=gt_edge_data,
            edge_ids=edge_ids,
            gt_od_vals=gt_od_vals,
            routes_df=routes_df,
            base_path=base_path,
            bounds=bounds,
            device=device,
            dtype=dtype,
            train_X_init=train_X_init_norm,
            train_Y_errors_init=train_Y_errors_init,
            n_restarts=n_restarts,
            n_bo_iterations=n_iterations,
            error_metric_func=error_metric_func,
            use_flow_proportional_weights = (args.error_metric == "cubic"),
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )




        #saving results 
   
        print("SAVING RESULTS")
      

        best_result = aggregated_results["overall_best_result"]
        print(f"\nBest result (restart {aggregated_results['overall_best_restart'] + 1}):")
        print(f"  Best S: {best_result.best_S:.6f}")
 
        # Save per-restart convergence (model/common) for later aggregation/plotting
        all_results = aggregated_results.get("all_results", [])
        if all_results:
            for i, res in enumerate(all_results, start=1):
                # Calculate seed for this restart (matches notebook: 42 + restart * 1000)
                restart_seed = 42 + (i - 1) * 1000  # restart 1 -> seed 42, restart 2 -> seed 1042, etc.
                subdir = path_result / f"restart_{i}_seed-{restart_seed}"
                subdir.mkdir(parents=True, exist_ok=True)

                conv_df = pd.DataFrame(
                    {
                        "iteration": range(len(res.convergence_curve)),
                        "best_S_model": res.convergence_curve,
                    }
                )
                conv_df.to_csv(subdir / "convergence.csv", index=False)

                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(7, 4.5))
                    plt.plot(conv_df["iteration"], conv_df["best_S_model"], marker="o", label="S (raw)")
                    plt.xlabel("Iteration")
                    plt.ylabel("Best S")
                    plt.title(f"Convergence: {network_name} - {model_name} (restart {i})")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.yscale("log")
                    plt.savefig(subdir / "convergence.png", dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"  Warning: could not plot convergence for restart {i}: {e}")


        # Save convergence data
        model_curve = best_result.convergence_curve
        convergence_df = pd.DataFrame({
            "iteration": range(len(model_curve)),
            "best_S_model": model_curve,
        })
        convergence_df.to_csv(path_result / "convergence.csv", index=False)
        
        # Also save best result as BOResults pickle for later loading
        from src.optimization.io import save_bo_results
        save_bo_results(
            results=best_result,
            save_dir=path_result,
            restart_idx=None,  # Best result, not a specific restart
            seed=None,
            metadata={
                "model_name": model_name,
                "kernel": kernel,
                "network_name": network_name,
                "best_restart": aggregated_results['overall_best_restart'] + 1,
            }
        )
        
        print(f"\nResults saved to: {path_result}")

        # Log best values
        print(f"  Best S (raw): {best_result.best_S:.6f}")

        # Result visualization: plot convergence curve
        print("\nGenerating convergence plot...")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            convergence_df["iteration"],
            convergence_df["best_S_model"],
            marker="o",
            label=f"{model_name} (S raw)",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Best S")
        plt.title(f"Convergence: {network_name} - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.savefig(path_result / "convergence.png", dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved to: {path_result / 'convergence.png'}")


if __name__ == "__main__":
    main()

