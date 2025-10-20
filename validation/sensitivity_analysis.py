# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Validation Script

This script performs a sensitivity analysis for the overall IRIS Total Score
to validate the robustness of the rankings against perturbations in the internal
weighting schemes.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.calculations import normalize_deviation_space


class ValConfig:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "final_evaluation")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "validation")


def calculate_total_score(df_normalized):
    distances = np.linalg.norm(df_normalized.values, axis=1)
    return 50000 * np.exp(-5 * distances)


def main():
    print("=" * 50)
    print("         Running Sensitivity Analysis Validation         ")
    print("=" * 50)
    os.makedirs(ValConfig.OUTPUT_DIR, exist_ok=True)

    try:
        raw_data_path = os.path.join(ValConfig.EVAL_RESULTS_DIR, "total_deviation_space_raw.csv")
        df_raw = pd.read_csv(raw_data_path, index_col='model')
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{raw_data_path}'.")
        print("Please run the main 'evaluate.py' script first to generate this file.")
        return

    baseline_normalized = normalize_deviation_space(df_raw.copy())
    baseline_scores = calculate_total_score(baseline_normalized)

    scenarios = {
        "Up-weight IFS": {'dims': [c for c in df_raw.columns if c.startswith(('RD_', 'AD_', 'SPD_'))], 'factor': 1.1},
        "Down-weight IFS": {'dims': [c for c in df_raw.columns if c.startswith(('RD_', 'AD_', 'SPD_'))], 'factor': 0.9},
        "Up-weight RFS": {'dims': [c for c in df_raw.columns if c.startswith(('JSD_', 'AbsSDS_'))], 'factor': 1.1},
        "Down-weight RFS": {'dims': [c for c in df_raw.columns if c.startswith(('JSD_', 'AbsSDS_'))], 'factor': 0.9},
        "Up-weight BIS": {'dims': [c for c in df_raw.columns if c.startswith(('Penalty_', 'ac_diff_', 'dhr_'))],
                          'factor': 1.1},
        "Down-weight BIS": {'dims': [c for c in df_raw.columns if c.startswith(('Penalty_', 'ac_diff_', 'dhr_'))],
                            'factor': 0.9},
    }

    results = []
    for name, config in tqdm(scenarios.items(), desc="Running Scenarios"):
        df_perturbed_raw = df_raw.copy()
        for dim in config['dims']:
            if dim in df_perturbed_raw.columns:
                df_perturbed_raw[dim] *= config['factor']

        df_perturbed_normalized = normalize_deviation_space(df_perturbed_raw)
        perturbed_scores = calculate_total_score(df_perturbed_normalized)

        rho, p_val = spearmanr(baseline_scores, perturbed_scores)
        results.append({
            'Scenario': name,
            'Spearman_Rho': rho,
            'p_value': p_val
        })

    df_results = pd.DataFrame(results)
    output_path = os.path.join(ValConfig.OUTPUT_DIR, "sensitivity_analysis_results.csv")
    df_results.to_csv(output_path, index=False, float_format="%.4f")

    print("\n--- Sensitivity Analysis Results ---")
    print(df_results.to_string())
    print(f"\nResults saved to '{output_path}'")


if __name__ == "__main__":
    main()
