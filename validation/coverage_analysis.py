# -*- coding: utf-8 -*-
"""
Coverage & Comprehensiveness Validation Script

This script validates the structural design of the IRIS benchmark by analyzing
the correlation between its different dimensions.
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.dimensions import (
    calculate_ifs_gen_submetrics, calculate_rfs_gen_submetrics,
    calculate_bis_gen_submetrics, calculate_ifs_und_submetrics,
    calculate_rfs_und_submetrics, calculate_bis_und_submetrics
)
from metrics.calculations import normalize_deviation_space


class ValConfig:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
    ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "validation")


def main():
    print("=" * 50)
    print("  Running Coverage & Comprehensiveness Validation  ")
    print("=" * 50)
    os.makedirs(ValConfig.OUTPUT_DIR, exist_ok=True)

    print("\n>>> Loading all sub-metrics from processed data...")
    tasks = {
        'IFS_Gen': calculate_ifs_gen_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "generation")),
        'RFS_Gen': calculate_rfs_gen_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "generation"),
                                                ValConfig.ANNOTATIONS_DIR),
        'BIS_Gen': calculate_bis_gen_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "bis_gen")),
        'IFS_Und': calculate_ifs_und_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "vqa"),
                                                ValConfig.ANNOTATIONS_DIR),
        'RFS_Und': calculate_rfs_und_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "vqa"),
                                                ValConfig.ANNOTATIONS_DIR),
        'BIS_Und': calculate_bis_und_submetrics(os.path.join(ValConfig.PROCESSED_DATA_DIR, "bis_und"),
                                                ValConfig.ANNOTATIONS_DIR)
    }

    all_dfs = [df for df in tasks.values() if not df.empty]
    if not all_dfs:
        print("No sub-metric data found. Aborting.")
        return

    df_total_raw = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'),
                          all_dfs).fillna(0)

    print("\n>>> Performing high-level dimension correlation analysis...")
    df_high_level = pd.DataFrame(index=df_total_raw.index)
    for task_name, df_task in tasks.items():
        if not df_task.empty:
            df_task_normalized = normalize_deviation_space(df_task)
            df_high_level[task_name] = np.linalg.norm(df_task_normalized.values, axis=1)

    high_level_corr = df_high_level.corr()
    high_level_corr.to_csv(os.path.join(ValConfig.OUTPUT_DIR, 'coverage_high_level_matrix.csv'))

    plt.figure(figsize=(10, 8))
    sns.heatmap(high_level_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('High-Level Dimension Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(ValConfig.OUTPUT_DIR, 'coverage_high_level_heatmap.png'), dpi=300)
    plt.close()
    print("High-level correlation matrix and heatmap saved.")

    print("\n>>> Coverage & Comprehensiveness Validation Complete.")


if __name__ == "__main__":
    main()
