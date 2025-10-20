# -*- coding: utf-8 -*-
"""
Internal Consistency Validation Script (Cronbach's Alpha)

This script validates the internal consistency of sub-metrics for each of the six
core fairness dimensions using Cronbach's Alpha.
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys

# Add project root to path to allow direct import from metrics module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.dimensions import (
    calculate_ifs_gen_submetrics, calculate_rfs_gen_submetrics,
    calculate_bis_gen_submetrics, calculate_ifs_und_submetrics,
    calculate_rfs_und_submetrics, calculate_bis_und_submetrics
)

# --- Configuration ---
# Assume the script is run from the root of the project `iris-benchmark/`
# The paths are relative to the project root.
class ValConfig:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
    ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "validation")

def cronbach_alpha(df: pd.DataFrame):
    """Calculates Cronbach's Alpha for a given DataFrame of items."""
    df = df.dropna(axis=1, how='all').dropna(axis=0)
    if df.shape[1] < 2 or df.shape[0] < 2:
        return np.nan

    k = df.shape[1]
    sum_item_variances = df.var(axis=0, ddof=1).sum()
    total_score_variance = df.sum(axis=1).var(ddof=1)
    if total_score_variance == 0: return np.nan
    alpha = (k / (k - 1)) * (1 - (sum_item_variances / total_score_variance))
    return alpha

def main():
    print("=" * 50)
    print("  Running Internal Consistency Validation (Cronbach's Alpha)  ")
    print("=" * 50)
    os.makedirs(ValConfig.OUTPUT_DIR, exist_ok=True)

    print("\n>>> Loading all sub-metrics from processed data...")
    tasks = {
        'IFS_Gen': calculate_ifs_gen_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "generation")),
        'RFS_Gen': calculate_rfs_gen_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "generation"), ValConfig.ANNOTATIONS_DIR),
        'BIS_Gen': calculate_bis_gen_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "bis_gen")),
        'IFS_Und': calculate_ifs_und_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "vqa"), ValConfig.ANNOTATIONS_DIR),
        'RFS_Und': calculate_rfs_und_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "vqa"), ValConfig.ANNOTATIONS_DIR),
        'BIS_Und': calculate_bis_und_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "bis_und"), ValConfig.ANNOTATIONS_DIR)
    }

    print("\n>>> Calculating Cronbach's Alpha for each fairness dimension...")
    results = []
    for task_name, df in tqdm(tasks.items(), desc="Calculating Alpha"):
        if df.empty:
            alpha, num_items, num_models = np.nan, 0, 0
        else:
            df_filtered = df.loc[:, (df.var() > 1e-9)] # Filter out columns with no variance
            alpha = cronbach_alpha(df_filtered)
            num_items = df_filtered.shape[1]
            num_models = df_filtered.shape[0]

        results.append({
            'Dimension': task_name,
            'Cronbach_Alpha': alpha,
            'Num_SubMetrics': num_items,
            'Num_Models': num_models
        })

    df_results = pd.DataFrame(results)
    output_path = os.path.join(ValConfig.OUTPUT_DIR, 'internal_consistency_results.csv')
    df_results.to_csv(output_path, index=False, float_format="%.4f")

    print("\n>>> Internal Consistency Validation Complete.")
    print(df_results)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
