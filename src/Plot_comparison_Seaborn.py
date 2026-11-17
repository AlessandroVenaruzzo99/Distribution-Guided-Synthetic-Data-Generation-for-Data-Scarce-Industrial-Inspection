import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_f1_score(precision, recall):
    """Calculates the F1-score."""
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def debug_csv_columns(csv_path):
    """Debug function to show available columns in CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"  Columns in {csv_path}:")
        print(f"  {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        if len(df) > 0:
            print(f"  First few rows:")
            print(df.head(3).to_string())
        return df.columns.tolist()
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return []


def find_metric_columns(df):
    """Maps raw training CSV columns to standardized metric names."""
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'precision' in col_lower:
            column_mapping['precision'] = col
        elif 'recall' in col_lower:
            column_mapping['recall'] = col
        elif (
            'map50-95' in col_lower
            or 'map@0.5:0.95' in col_lower
            or 'map_50_95' in col_lower
            or ('map50' in col_lower and '95' in col_lower)
        ):
            column_mapping['mAP50-95'] = col
        elif 'map50' in col_lower or 'map@0.5' in col_lower or 'map_50' in col_lower:
            column_mapping['mAP50'] = col
    return column_mapping


def save_fold_summary_with_mean_std(baseline_name, results_folder, fold_metric_rows):
    """Saves fold metrics plus mean and std rows in the same CSV format used for reporting."""
    if not fold_metric_rows:
        print(f"Warning: No fold summary rows available for baseline '{baseline_name}'.")
        return

    summary_df = pd.DataFrame(fold_metric_rows)
    metric_columns = [c for c in ['mp', 'mr', 'mf1', 'map_IoU[50]', 'map_IoU[50-95]'] if c in summary_df.columns]

    mean_row = {'folds': 'mean'}
    std_row = {'folds': 'std'}
    for metric in metric_columns:
        mean_row[metric] = summary_df[metric].mean()
        std_row[metric] = summary_df[metric].std(ddof=1)

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    summary_csv_path = os.path.join(results_folder, "mean_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary metrics with mean/std: {summary_csv_path}")


def _parse_metric_list(value):
    """Parses list-like strings such as '[0.1 0.2 0.3]' into float lists."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.strip('[]')
    if not text:
        return []

    # Supports both space-separated and comma-separated numeric lists.
    tokens = re.split(r"[\s,]+", text)
    parsed = []
    for token in tokens:
        if not token:
            continue
        try:
            parsed.append(float(token))
        except ValueError:
            return None
    return parsed


def _format_metric_list(values, precision=6):
    """Formats a list of floats in the same bracketed style used in CSV reports."""
    if values is None:
        return ""
    return "[" + " ".join(f"{v:.{precision}f}" for v in values) + "]"


def save_class_summary_with_mean_std(results_folder):
    """Appends mean and std rows (per class) to class_metrics.csv."""
    class_csv_path = os.path.join(results_folder, "class_metrics.csv")
    if not os.path.exists(class_csv_path):
        print(f"Warning: class_metrics.csv not found in {results_folder}. Skipping class std computation.")
        return

    try:
        class_df = pd.read_csv(class_csv_path)
    except Exception as e:
        print(f"Warning: Could not read class_metrics.csv in {results_folder}: {e}")
        return

    if 'folds' not in class_df.columns:
        print(f"Warning: 'folds' column missing in {class_csv_path}. Skipping class std computation.")
        return

    fold_rows = class_df[class_df['folds'].astype(str).str.lower().str.startswith('fold')].copy()
    if fold_rows.empty:
        print(f"Warning: No fold rows found in {class_csv_path}. Skipping class std computation.")
        return

    class_metric_cols = [
        c for c in ['cl_ap', 'cl_ar', 'cl_af1', 'cl_ap[50]', 'cl_ap_IoU[50-95]']
        if c in fold_rows.columns
    ]

    mean_row = {col: "" for col in class_df.columns}
    std_row = {col: "" for col in class_df.columns}
    mean_row['folds'] = 'mean'
    std_row['folds'] = 'std'

    for metric_col in class_metric_cols:
        parsed_lists = fold_rows[metric_col].apply(_parse_metric_list).tolist()
        parsed_lists = [vals for vals in parsed_lists if vals is not None]
        if not parsed_lists:
            continue

        class_count = len(parsed_lists[0])
        if class_count == 0 or any(len(vals) != class_count for vals in parsed_lists):
            print(
                f"Warning: Inconsistent class vector lengths in {metric_col} for {class_csv_path}. Skipping this metric."
            )
            continue

        per_class_df = pd.DataFrame(parsed_lists)
        mean_vals = per_class_df.mean(axis=0).tolist()
        std_vals = per_class_df.std(axis=0, ddof=1).fillna(0).tolist()

        mean_row[metric_col] = _format_metric_list(mean_vals)
        std_row[metric_col] = _format_metric_list(std_vals)

    output_df = pd.concat(
        [fold_rows, pd.DataFrame([mean_row, std_row], columns=class_df.columns)],
        ignore_index=True,
    )
    output_df.to_csv(class_csv_path, index=False)
    print(f"Saved class metrics with mean/std: {class_csv_path}")

def analyze_baselines_mean_with_variance(base_folder_path, debug_mode=True, ci=('sd', 1)):
    """
    Analyzes baseline results across ALL folds and generates plots showing
    mean values with variance (confidence intervals) using seaborn.

    Args:
        base_folder_path (str): The absolute path to the "Baseline Comparison" folder.
        debug_mode (bool): If True, shows column information for debugging.
        ci (tuple or int): Confidence interval type. Options:
            - ('sd', 1): ±1 Standard deviation
            - ('sd', 2): ±2 Standard deviations
            - ('se', 1): ±1 Standard error
            - ('ci', 95): 95% confidence interval
            - ('pi', 95): 95% percentile interval
    """
    if not os.path.exists(base_folder_path) or not os.path.isdir(base_folder_path):
        print(f"Error: The specified base folder '{base_folder_path}' does not exist or is not a directory.")
        return

    baseline_paths = [os.path.join(base_folder_path, b) for b in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, b))]

    # Collect all fold data into a single DataFrame for seaborn
    all_data = []

    for baseline_path in baseline_paths:
        baseline_name = os.path.basename(baseline_path)
        print(f"Analyzing baseline: {baseline_name}")

        results_folder = os.path.join(baseline_path, "results")
        if not os.path.exists(results_folder):
            print(f"Warning: 'results' folder not found in {baseline_path}. Skipping.")
            continue

        fold_folders = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.startswith("fold") and os.path.isdir(os.path.join(results_folder, f))]
        baseline_fold_metrics = []

        for fold_path in fold_folders:
            fold_name = os.path.basename(fold_path)
            fold_number = fold_name.split('_')[0].replace('fold', '')
            yolo_train_folder_name = f"yolo_train_{fold_number}"
            
            yolo_train_path = os.path.join(fold_path, "train", yolo_train_folder_name)
            results_csv_path = os.path.join(yolo_train_path, "results.csv")

            if not os.path.exists(results_csv_path):
                print(f"Warning: results.csv not found in {yolo_train_path}. Skipping fold '{fold_name}'.")
                continue

            if debug_mode:
                available_columns = debug_csv_columns(results_csv_path)
                debug_mode = False
                
            df = pd.read_csv(results_csv_path)

            # Find column mappings
            column_mapping = find_metric_columns(df)

            required_cols = ['precision', 'recall', 'mAP50-95']
            missing_cols = [col for col in required_cols if col not in column_mapping]
            
            if missing_cols:
                print(f"Warning: Could not find columns for {missing_cols} in {results_csv_path}. Skipping fold '{fold_name}'.")
                continue

            # Create standardized dataframe
            fold_data = pd.DataFrame({
                'epoch': range(1, len(df) + 1),
                'precision': df[column_mapping['precision']].values,
                'recall': df[column_mapping['recall']].values,
                'mAP50-95': df[column_mapping['mAP50-95']].values,
                'baseline': baseline_name,
                'fold': fold_name
            })
            
            # Calculate F1-score
            fold_data['f1-score'] = fold_data.apply(
                lambda row: calculate_f1_score(row['precision'], row['recall']), axis=1
            )

            # Collect fold-level summary from final epoch to build mean/std report CSV.
            last_epoch = fold_data.iloc[-1]
            summary_row = {
                'folds': f'fold {fold_number}',
                'mp': last_epoch['precision'],
                'mr': last_epoch['recall'],
                'mf1': last_epoch['f1-score'],
                'map_IoU[50-95]': last_epoch['mAP50-95']
            }
            if 'mAP50' in column_mapping:
                summary_row['map_IoU[50]'] = df[column_mapping['mAP50']].iloc[-1]
            baseline_fold_metrics.append(summary_row)
            
            all_data.append(fold_data)
            print(f"  Loaded {len(fold_data)} epochs from {fold_name}")

        save_fold_summary_with_mean_std(baseline_name, results_folder, baseline_fold_metrics)
        save_class_summary_with_mean_std(results_folder)

    if not all_data:
        print("No valid baseline results found to plot.")
        return

    # Combine all data into single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data points: {len(combined_df)}")
    print(f"Baselines: {combined_df['baseline'].unique()}")
    print(f"Folds per baseline: {combined_df.groupby('baseline')['fold'].nunique().to_dict()}")

    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Generate plots for all metrics
    metrics_to_plot = ['precision', 'recall', 'f1-score', 'mAP50-95']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        
        # Use seaborn lineplot - automatically calculates mean and CI across folds
        sns.lineplot(
            data=combined_df,
            x='epoch',
            y=metric,
            hue='baseline',
            errorbar=ci,  # 'sd' for standard deviation, 95 for 95% CI, None for no CI
            linewidth=2
        )
        
        # Update title to reflect the actual errorbar type
        plt.title(f'{metric.capitalize()} Over Epochs (Mean ± {"SD" if ci == "sd" else f"{ci}% CI"} Across All Folds)')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.xlim(1, 50)  # Start at 1, extra margin around 50
        plt.ylim(0, 1.05)  # Start at 0, extra margin around 1
        plt.margins(x=0)  # Remove automatic x-axis margins
        plt.legend(title="Baselines", loc='lower right', framealpha=0.9, fancybox=True)
        plt.tight_layout()
        
        plot_filename = os.path.join(base_folder_path, f'{metric}_mean_variance_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Generated plot: {plot_filename}")
        plt.close()
    
    # Save the combined data for reference
    combined_csv_path = os.path.join(base_folder_path, "all_folds_combined_data.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined data saved to: {combined_csv_path}")
    
    print("Analysis complete (mean with variance across all folds).")
    
    return combined_df


def analyze_baselines_mean_with_variance_odd_epochs(base_folder_path, debug_mode=True, ci='sd'):
    """
    Same as analyze_baselines_mean_with_variance but using only odd epochs.

    Args:
        base_folder_path (str): The absolute path to the "Baseline Comparison" folder.
        debug_mode (bool): If True, shows column information for debugging.
        ci (str or int): Confidence interval type ('sd', 95, or None).
    """
    if not os.path.exists(base_folder_path) or not os.path.isdir(base_folder_path):
        print(f"Error: The specified base folder '{base_folder_path}' does not exist or is not a directory.")
        return

    baseline_paths = [os.path.join(base_folder_path, b) for b in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, b))]

    all_data = []

    for baseline_path in baseline_paths:
        baseline_name = os.path.basename(baseline_path)
        print(f"Analyzing baseline: {baseline_name}")

        results_folder = os.path.join(baseline_path, "results")
        if not os.path.exists(results_folder):
            print(f"Warning: 'results' folder not found in {baseline_path}. Skipping.")
            continue

        fold_folders = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.startswith("fold") and os.path.isdir(os.path.join(results_folder, f))]
        baseline_fold_metrics = []

        for fold_path in fold_folders:
            fold_name = os.path.basename(fold_path)
            fold_number = fold_name.split('_')[0].replace('fold', '')
            yolo_train_folder_name = f"yolo_train_{fold_number}"
            
            yolo_train_path = os.path.join(fold_path, "train", yolo_train_folder_name)
            results_csv_path = os.path.join(yolo_train_path, "results.csv")

            if not os.path.exists(results_csv_path):
                continue

            if debug_mode:
                debug_csv_columns(results_csv_path)
                debug_mode = False
                
            df = pd.read_csv(results_csv_path)

            column_mapping = find_metric_columns(df)

            required_cols = ['precision', 'recall', 'mAP50-95']
            if any(col not in column_mapping for col in required_cols):
                continue

            # Filter to odd epochs only (indices 0, 2, 4, ... correspond to epochs 1, 3, 5, ...)
            df_odd = df.iloc[::2].copy()
            
            fold_data = pd.DataFrame({
                'epoch': [2*i + 1 for i in range(len(df_odd))],  # 1, 3, 5, 7, ...
                'precision': df_odd[column_mapping['precision']].values,
                'recall': df_odd[column_mapping['recall']].values,
                'mAP50-95': df_odd[column_mapping['mAP50-95']].values,
                'baseline': baseline_name,
                'fold': fold_name
            })
            
            fold_data['f1-score'] = fold_data.apply(
                lambda row: calculate_f1_score(row['precision'], row['recall']), axis=1
            )

            # Keep a fold-level report based on the last odd epoch used in this analysis.
            last_epoch = fold_data.iloc[-1]
            summary_row = {
                'folds': f'fold {fold_number}',
                'mp': last_epoch['precision'],
                'mr': last_epoch['recall'],
                'mf1': last_epoch['f1-score'],
                'map_IoU[50-95]': last_epoch['mAP50-95']
            }
            if 'mAP50' in column_mapping:
                summary_row['map_IoU[50]'] = df_odd[column_mapping['mAP50']].iloc[-1]
            baseline_fold_metrics.append(summary_row)
            
            all_data.append(fold_data)

        save_fold_summary_with_mean_std(baseline_name, results_folder, baseline_fold_metrics)
        save_class_summary_with_mean_std(results_folder)

    if not all_data:
        print("No valid baseline results found to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    sns.set_style("whitegrid")
    metrics_to_plot = ['precision', 'recall', 'f1-score', 'mAP50-95']

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        plt.rcParams.update({'font.size': 16})  # Increase font size for better readability
        
        sns.lineplot(
            data=combined_df,
            x='epoch',
            y=metric,
            hue='baseline',
            errorbar=('sd', 1),  
            linewidth=2,
            marker='o',
            markersize=4
        )
        
        plt.title(f'{metric.capitalize()} Over Epochs (Mean ± {"SD" if ci == "sd" else f"{ci}% CI"} Across All Folds)')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.xlim(0, 50)  # Start at 0, extra margin around 50
        plt.ylim(0, 1.05)  # Start at 0, extra margin around 1
        plt.legend(title="Baselines", loc='lower right', framealpha=0.9, fancybox=True)
        plt.tight_layout()
        
        plot_filename = os.path.join(base_folder_path, f'{metric}_mean_variance_odd_epochs_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Generated plot: {plot_filename}")
        plt.close()
    
    print("Analysis complete (mean with variance, odd epochs only).")
    
    return combined_df


if __name__ == "__main__":
    import argparse

    # Expected layout: <folder>/<baseline_name>/results/fold*_CV/train/... (one sub-folder per compared method)
    parser = argparse.ArgumentParser(description="Plot mean +/- SD training curves across CV folds for several baselines")
    parser.add_argument("--baselines_dir", type=str, required=True,
                        help="Folder containing one sub-folder per baseline / ablation configuration")
    parser.add_argument("--ci", type=str, default="sd", help="'sd' for standard deviation bands or an integer for a CI (e.g. 95)")
    args = parser.parse_args()
    ci = args.ci if args.ci == "sd" else int(args.ci)

    # Mean with variance (standard deviation) across all folds
    analyze_baselines_mean_with_variance(args.baselines_dir, debug_mode=True, ci=ci)

    # Mean with variance using odd epochs only
    analyze_baselines_mean_with_variance_odd_epochs(args.baselines_dir, debug_mode=True, ci=ci)
