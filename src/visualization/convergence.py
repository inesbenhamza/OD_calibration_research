"""
Convergence and CSV-based visualization functions.

This module contains functions for post-hoc analysis of BO results from CSV files.
It reads convergence.csv files and creates convergence plots and fit-to-GT plots.

This is separate from bo_results.py which works with BOResults objects in-memory.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os


#plot fit to ground truth
def plot_fitGT(
    eval_measure,
    network_name,
    model_seed_dict,
    data_sets,
    sensor_measure_simul,
    gt_csv,
    fig_path
):
    if eval_measure == "count":
        col_name = "interval_nVehContrib"
    elif eval_measure == "speed":
        col_name = "interval_harmonicMeanSpeed"
    else:
        raise ValueError(f"Invalid eval_measure: {eval_measure}. Please provide one of ['count', 'speed'].")
    
    # FIX: Handle both 'epoch' and 'iteration' columns, and 'loss' vs 'best_S_model'
    data_sets_for_fitPlot = {}
    for key, df in data_sets.items():
        if 'epoch' in df.columns:
            filtered = df[df['epoch'] != 0]
        elif 'iteration' in df.columns:
            filtered = df[df['iteration'] != 0]
        else:
            continue  # Skip if neither column exists
        if len(filtered) > 0:
            data_sets_for_fitPlot[key] = filtered

    for model, seeds in model_seed_dict.items():
        plt.figure(figsize=(7, 5))
        combined_data = []

        for seed in seeds:
            key = (model, seed)
            if key not in data_sets_for_fitPlot:
                continue
                
            df = data_sets_for_fitPlot[key]
            
            # FIX: Handle both 'loss' and 'best_S_model' columns
            if 'loss' in df.columns:
                min_loss_row = df.loc[df['loss'].idxmin()]
                loss_col = 'loss'
            elif 'best_S_model' in df.columns:
                min_loss_row = df.loc[df['best_S_model'].idxmin()]
                loss_col = 'best_S_model'
            else:
                print(f"Warning: No loss column found for {key}, skipping fitGT plot")
                continue
            
            # FIX: Handle both 'epoch' and 'iteration' columns
            if 'epoch' in min_loss_row:
                min_epoch = int(min_loss_row['epoch'])
            elif 'iteration' in min_loss_row:
                min_epoch = int(min_loss_row['iteration'])
            else:
                continue
                
            # FIX: Handle 'batch' column (may not exist)
            if 'batch' in min_loss_row:
                min_batch = int(min_loss_row['batch'])
            else:
                min_batch = 0  # Default if no batch column

            # FIX: Check if sensor_measure_simul has data
            if key not in sensor_measure_simul or sensor_measure_simul[key] is None:
                print(f"Warning: No sensor data for {key}, skipping fitGT plot")
                continue
                
            sensor_data = sensor_measure_simul[key]
            
            # FIX: Handle both 'epoch' and 'iteration' in sensor data
            if 'epoch' in sensor_data.columns:
                filtered_data = sensor_data[
                    (sensor_data['epoch'] == min_epoch) & 
                    (sensor_data.get('batch', 0) == min_batch if 'batch' in sensor_data.columns else True)
                ]
            elif 'iteration' in sensor_data.columns:
                filtered_data = sensor_data[
                    (sensor_data['iteration'] == min_epoch) & 
                    (sensor_data.get('batch', 0) == min_batch if 'batch' in sensor_data.columns else True)
                ]
            else:
                # If no epoch/iteration, use all data
                filtered_data = sensor_data
                
            if len(filtered_data) > 0:
                combined_data.append(filtered_data)

        if not combined_data:
            print(f"Warning: No data to plot for {model}, skipping fitGT plot")
            plt.close()
            continue
            
        combined_df = pd.concat(combined_data)

        mean_values = combined_df.groupby('link_id')[col_name].mean().reset_index()
        std_values = combined_df.groupby('link_id')[col_name].std().reset_index()

        merged_mean_df = gt_csv.merge(mean_values, on='link_id', suffixes=('_gt', '_simul'))
        merged_mean_df = merged_mean_df.merge(std_values.rename(columns={col_name: 'std'}), on='link_id')

        max_gt_value = max(merged_mean_df[col_name + '_gt'].max(), merged_mean_df[col_name + '_simul'].max())
        min_gt_value = min(merged_mean_df[col_name + '_gt'].min(), merged_mean_df[col_name + '_simul'].min())

        plt.plot([min_gt_value, max_gt_value], [min_gt_value, max_gt_value], 'r-', label="45-degree line")
  
        plt.scatter(
            merged_mean_df[col_name + '_gt'],
            merged_mean_df[col_name + '_simul'],
            label=f"{model}",
            alpha=0.7
        )
        
        plt.errorbar(
            merged_mean_df[col_name + '_gt'],
            merged_mean_df[col_name + '_simul'],
            yerr=merged_mean_df['std'] / 2,
            fmt='o',
            alpha=1.0,
            label=f"{model} error",
            color='black'
        )

        if model == 'vanillabo':
            title_name = 'Vanilla BO'
        elif model == 'independent_gp':
            title_name = 'Independent GP'
        elif model == 'mogp':
            title_name = 'Mogp'
        else:
            raise ValueError(f"Invalid model: {model}. Please provide one of ['independent_gp', 'vanillabo', 'mogp'].")

        plt.xlabel(f"GT link measurements ({eval_measure})", fontsize=20)
        plt.ylabel(f"Simulated link measurements ({eval_measure})", fontsize=20)
        plt.title(f"Fit to GT | {title_name}", fontsize=16)
        plt.savefig(os.path.join(fig_path, f'FitGT_{network_name}_{model}.png'), dpi=300)
        plt.close()


#plot two convergence curves
def plot_two_convergence(csv_a, label_a, csv_b, label_b, out_path, title=None, use_common=False):
    """
    Overlay two convergence curves.
    - use_common=False plots best_S_model
    - use_common=True  plots best_S_common
    """
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    key = "best_S_common" if use_common else "best_S_model"
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_a["iteration"], df_a[key], marker="o", label=label_a)
    plt.plot(df_b["iteration"], df_b[key], marker="o", label=label_b)
    plt.xlabel("Iteration")
    plt.ylabel("Best S")
    plt.title(title or "Convergence comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_restarts_with_mean(model_runs, out_path, title=None):
    """
    Plot all restart curves (faint) and the mean curve for each model.
    Uses best_S_model; log-y.

    Parameters
    ----------
    model_runs : dict[str, list[str]]
        Mapping model_name -> list of convergence.csv paths (one per restart).
    out_path : str
        Output PNG path.
    title : str, optional
        Figure title.
    """
    plt.figure(figsize=(7, 5))

    for model, csv_list in model_runs.items():
        if not csv_list:
            continue
        dfs = [pd.read_csv(p) for p in csv_list]
        # Align to shortest length
        T = min(len(df) for df in dfs)
        curves = []
        for df in dfs:
            c = df["best_S_model"].values[:T]
            curves.append(c)
        curves = np.stack(curves, axis=0)  # [n_restarts, T]
        iters = np.arange(T)

        # Plot each restart (faint)
        for c in curves:
            plt.plot(iters, c, color="gray", alpha=0.25)

        # Plot mean
        mean = curves.mean(axis=0)
        plt.plot(iters, mean, label=model, linewidth=2)

    plt.yscale("log")
    if title:
        plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best S")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_restarts_mean_only(model_runs, out_path, title=None, use_common=False):
    """
    Plot only the mean restart curve for each model (no individual curves).

    Parameters
    ----------
    model_runs : dict[str, list[str]]
        Mapping model_name -> list of convergence.csv paths (one per restart).
    out_path : str
        Output PNG path.
    title : str, optional
        Figure title.
    use_common : bool
        If True, use 'best_S_common'; otherwise use 'best_S_model'.
    """
    plt.figure(figsize=(7, 5))
    key = "best_S_common" if use_common else "best_S_model"

    for model, csv_list in model_runs.items():
        if not csv_list:
            continue
        dfs = [pd.read_csv(p) for p in csv_list]
        T = min(len(df) for df in dfs)
        curves = []
        for df in dfs:
            c = df[key].values[:T]
            curves.append(c)
        curves = np.stack(curves, axis=0)  # [n_restarts, T]
        iters = np.arange(T)

        mean = curves.mean(axis=0)
        plt.plot(iters, mean, label=model, linewidth=2)

    plt.yscale("log")
    if title:
        plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best S")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_restarts_mean_with_ci(
    model_runs,
    out_path,
    title=None,
    use_common=False,
    ci="std",
    ci_mult=1.0,
    align="shortest",
):
    """
    Plot mean restart curve per model with a shaded confidence band.

    Parameters
    ----------
    model_runs : dict[str, list[str]]
        Mapping model_name -> list of convergence.csv paths (one per restart).
    out_path : str
        Output PNG path.
    title : str, optional
        Figure title.
    use_common : bool
        If True, use 'best_S_common'; otherwise use 'best_S_model'.
    ci : {"std", None}
        Type of confidence band. "std" plots mean ± ci_mult * std. None plots no band.
    ci_mult : float
        Multiplier for the std band.
    align : {"shortest", "longest"}
        How to align runs:
        - "shortest": truncate all runs to the shortest length (default).
        - "longest": pad shorter runs with their last value to the longest length.
    """
    plt.figure(figsize=(7, 5))
    key = "best_S_common" if use_common else "best_S_model"

    for model, csv_list in model_runs.items():
        if not csv_list:
            continue
        dfs = [pd.read_csv(p) for p in csv_list]

        if align == "longest":
            T = max(len(df) for df in dfs)
            curves = []
            for df in dfs:
                series = df[key].values
                if len(series) < T:
                    # pad with last value
                    pad = np.full(T - len(series), series[-1])
                    series = np.concatenate([series, pad])
                curves.append(series)
        else:
            # shortest
            T = min(len(df) for df in dfs)
            curves = [df[key].values[:T] for df in dfs]

        curves = np.stack(curves, axis=0)  # [n_restarts, T]
        iters = np.arange(T)

        mean = curves.mean(axis=0)
        plt.plot(iters, mean, label=model, linewidth=2)

        if ci == "std":
            std = curves.std(axis=0)
            lower = mean - ci_mult * std
            upper = mean + ci_mult * std
            plt.fill_between(iters, lower, upper, alpha=0.15)

    plt.yscale("log")
    if title:
        plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best S")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_convergence(network_name, model_seed_dict, data_sets, fig_path):
    """
    Plot convergence curves for all models using data from convergence.csv files.
    
    Parameters
    ----------
    network_name : str
        Name of the network (e.g., "quickstart")
    model_seed_dict : dict
        Mapping model_name -> list of seeds (e.g., {'vanillabo': ['42', '1042']})
    data_sets : dict
        Mapping (model, seed) -> DataFrame with convergence data
    fig_path : str
        Directory to save the plot
    """
    # FIX: Remove dead code, work directly with DataFrames
    if not data_sets:
        print("Warning: No convergence data found, skipping convergence plot")
        return
        
    plt.figure(figsize=(7, 5))
    
    for model, seeds in model_seed_dict.items():
        curves = []
        for seed in seeds:
            key = (model, seed)
            if key in data_sets:
                df = data_sets[key]
                
                # Handle both 'iteration' and 'epoch' columns
                if 'iteration' in df.columns:
                    iters = df['iteration'].values
                    if 'best_S_model' in df.columns:
                        values = df['best_S_model'].values
                    elif 'best_S_common' in df.columns:
                        values = df['best_S_common'].values
                    else:
                        continue
                elif 'epoch' in df.columns:
                    iters = df['epoch'].values
                    # If there's a 'loss' column, use cummin
                    if 'loss' in df.columns:
                        values = df['loss'].cummin().values
                    elif 'best_S_model' in df.columns:
                        values = df['best_S_model'].values
                    elif 'best_S_common' in df.columns:
                        values = df['best_S_common'].values
                    else:
                        continue
                else:
                    continue
                curves.append((iters, values))
        
        if curves:
            # Align to shortest length
            min_len = min(len(v) for _, v in curves)
            aligned_curves = []
            aligned_iters = None
            
            for iters, values in curves:
                if aligned_iters is None:
                    aligned_iters = iters[:min_len]
                aligned_curves.append(values[:min_len])
            
            aligned_curves = np.stack(aligned_curves, axis=0)  # [n_restarts, T]
            
            # Plot each restart (faint)
            for c in aligned_curves:
                plt.plot(aligned_iters, c, color="gray", alpha=0.25)
            
            # Plot mean
            mean = aligned_curves.mean(axis=0)
            plt.plot(aligned_iters, mean, label=model, linewidth=2)
    
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best S")
    plt.title(f"Convergence: {network_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(fig_path, f'Convergence_{network_name}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


def main():
    """
    Main function to plot convergence and fit to ground truth from CSV files.
    This is a command-line script for post-hoc analysis of BO results.
    """
    #path 
    current_path = os.getcwd()
    if os.path.basename(current_path) == "visualization":
        current_path = os.path.dirname(current_path)
    current_path = current_path.replace("\\", "/")

    base_path_new = os.path.join(current_path, "output")
    fig_path = os.path.join(current_path, "visualization", "figures")
    sensor_path = os.path.join(current_path, "sensor_data")

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    parser = argparse.ArgumentParser(description="Plot figures")
    parser.add_argument(
        "--network_name",
        type=str,
        default="quickstart",
        choices=["2corridor","quickstart"],
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="matern-2p5",
        choices=["matern-1p5", "matern-2p5", "rbf"],
        help="GP kernel type used in the BO model",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="221014",
        help="Date for simulation",
    )
    parser.add_argument(
        "--hour",
        type=str,
        default="08-09",
        choices=["06-07", "08-09", "17-18"],
        help="Time for simulation",
    )
    parser.add_argument(
        "--eval_measure",
        type=str,
        default="count",
        #choices=["count", "speed"],
        help="Evaluation measurements"
    )
    parser.add_argument(
        "--routes_per_od",
        type=str,
        default='single',
        choices=["single", "multiple"],
        help="Type of routes to use for the simulation",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=3,
        help="Maximum number of epochs for simulation",
    )  
    
    args = parser.parse_args()
    print(args)
    
    network_name = args.network_name
    kernel = args.kernel
    date = args.date
    hour = args.hour
    eval_measure = args.eval_measure
    routes_per_od = args.routes_per_od
    max_epoch = args.max_epoch
    
    # FIX: Actually scan the output directory!
    from glob import glob
    pattern = os.path.join(base_path_new, f"{network_name}_*")
    all_folders = [os.path.basename(f) for f in glob(pattern) if os.path.isdir(f)]
    
    # Filter by kernel
    if kernel and kernel != "none":
        list_folder_name = [f for f in all_folders if kernel in f]
    else:
        list_folder_name = all_folders
    
    # Remove initSearch folders
    list_folder_name = [f for f in list_folder_name if 'initSearch' not in f]
    
    print(f"Found {len(all_folders)} folders matching {network_name}_*")
    print("Selected folders:", list_folder_name)
    
    # Parse model and seed from folder names (new structure: quickstart_vanillabo_matern-2p5)
    net_name = f'{network_name}_'
    model_list = []
    seeds_list = []
    valid_folders = []
    
    for folder in list_folder_name:
        if net_name not in folder:
            continue
        parts = folder.split('_')
        if len(parts) < 3:
            continue
        model = parts[1]  # e.g., "vanillabo"
        
        # Look for restart subdirectories in results/
        results_dir = os.path.join(base_path_new, folder, "results")
        if not os.path.exists(results_dir):
            continue
        
        # Find all restart_X_seed-Y subdirectories
        restart_dirs = [d for d in os.listdir(results_dir) 
                        if os.path.isdir(os.path.join(results_dir, d)) 
                        and d.startswith('restart_')]
        
        if not restart_dirs:
            # Fallback: check for single convergence.csv in results/
            conv_file = os.path.join(results_dir, "convergence.csv")
            if os.path.exists(conv_file):
                seed = "default"
                model_list.append(model)
                seeds_list.append(seed)
                valid_folders.append(folder)
        else:
            # Process each restart
            for restart_dir in restart_dirs:
                # Extract seed from directory name (restart_X_seed-Y)
                if 'seed-' in restart_dir:
                    seed = restart_dir.split('seed-')[1]
                else:
                    seed = "default"
                
                conv_file = os.path.join(results_dir, restart_dir, "convergence.csv")
                if os.path.exists(conv_file):
                    model_list.append(model)
                    seeds_list.append(seed)
                    valid_folders.append((folder, restart_dir))
    
    # Create mapping for data loading
    all_data = pd.DataFrame({
        'Folder': [f[0] if isinstance(f, tuple) else f for f in valid_folders],
        'RestartDir': [f[1] if isinstance(f, tuple) else None for f in valid_folders],
        'Model': model_list,
        'Seed': seeds_list
    })
    
    model_seed_list = all_data[['Folder', 'RestartDir', 'Model', 'Seed']].values.tolist()
    
    # Load convergence data from restart subdirectories
    data_sets = {}
    for file_name, restart_dir, model, seed in model_seed_list:
        if restart_dir:
            conv_file = os.path.join(base_path_new, file_name, "results", restart_dir, "convergence.csv")
        else:
            conv_file = os.path.join(base_path_new, file_name, "results", "convergence.csv")
        
        if os.path.exists(conv_file):
            df = pd.read_csv(conv_file)
            if 'epoch' in df.columns:
                df = df[df['epoch'] <= max_epoch]
            elif 'iteration' in df.columns:
                df = df[df['iteration'] <= max_epoch]
            data_sets[(model, seed)] = df
    
    model_seed_dict = all_data.groupby('Model')['Seed'].apply(list).to_dict()
    
    # Note: sensor_measure_simul might not exist in new structure
    sensor_measure_simul = {}
    
    # Plot convergence
    plot_convergence(
        network_name,
        model_seed_dict,
        data_sets,
        fig_path
    )
    
    # Fit to GT plot (skip if no sensor data)
    if sensor_measure_simul:
        try:
            gt_csv = pd.read_csv(f'{sensor_path}/{date}/gt_link_data_{network_name}_{date}_{hour}.csv')
            plot_fitGT(
                eval_measure,
                network_name,
                model_seed_dict,
                data_sets,
                sensor_measure_simul,
                gt_csv,
                fig_path
            )
        except Exception as e:
            print(f"Warning: Could not create fitGT plot: {e}")
    
    print("All plots have been generated successfully.")


if __name__ == "__main__":
    main()
