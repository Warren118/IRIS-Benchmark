# -*- coding: utf-8 -*-
"""
Impartiality Check Validation Script (Bias & Fairness Check)

This script performs a group analysis to check if the benchmark itself exhibits
any systematic bias towards certain types of models, focusing on architecture.
"""
import pandas as pd
import os
from scipy.stats import ttest_ind
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ValConfig:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "final_evaluation")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "validation")


# Define model metadata based on architecture from your paper
MODEL_METADATA = {
    'Bagel': {'type': 'UMLLM', 'structure': 'AR+Diff'},
    'BLIP3-0': {'type': 'UMLLM', 'structure': 'AR+Diff'},
    'Harmon': {'type': 'UMLLM', 'structure': 'AR'},
    'Janus-Pro': {'type': 'UMLLM', 'structure': 'AR'},
    'Show-o': {'type': 'UMLLM', 'structure': 'AR+Diff'},
    'UniWorld-V1': {'type': 'UMLLM', 'structure': 'AR+Diff'},
    'VILA-U': {'type': 'UMLLM', 'structure': 'AR+Diff'},
    # Add other models if they were included in the final scores
}


def main():
    print("=" * 50)
    print("      Running Benchmark Impartiality Check (t-test)      ")
    print("=" * 50)
    os.makedirs(ValConfig.OUTPUT_DIR, exist_ok=True)

    scores_path = os.path.join(ValConfig.EVAL_RESULTS_DIR, "final_iris_scores.csv")
    try:
        df_scores = pd.read_csv(scores_path)
    except FileNotFoundError:
        print(f"\nError: Final scores file not found at '{scores_path}'.")
        print("Please run 'evaluate.py' first.")
        return

    # Add metadata to the scores dataframe
    df_scores['structure'] = df_scores['model'].map(lambda x: MODEL_METADATA.get(x, {}).get('structure'))
    df_scores.dropna(subset=['structure'], inplace=True)

    print("\n>>> Performing group analysis based on architecture within UMLLMs...")
    ar_diff_scores = df_scores[df_scores['structure'] == 'AR+Diff']['IRIS_Total_Score']
    ar_scores = df_scores[df_scores['structure'] == 'AR']['IRIS_Total_Score']

    results = []
    if len(ar_diff_scores) > 1 and len(ar_scores) > 1:
        stat, p = ttest_ind(ar_diff_scores, ar_scores, equal_var=False)  # Welch's t-test
        results.append({
            'Test': 'AR+Diff vs. AR (within UMLLMs)',
            'T-statistic': stat,
            'p-value': p,
            'Group1_Mean': ar_diff_scores.mean(),
            'Group2_Mean': ar_scores.mean(),
        })
        print(f"\nWelch's t-test between AR+Diff and AR architectures:")
        print(f"  T-statistic: {stat:.4f}, p-value: {p:.4f}")
        if p > 0.05:
            print("  Conclusion: No significant difference found between architectures (p > 0.05).")
        else:
            print("  Conclusion: A significant difference was found between architectures (p <= 0.05).")
    else:
        print("\nCould not perform t-test: not enough models in each architecture group.")

    df_results = pd.DataFrame(results)
    output_path = os.path.join(ValConfig.OUTPUT_DIR, 'impartiality_check_results.csv')
    df_results.to_csv(output_path, index=False, float_format="%.4f")

    print("\n>>> Impartiality Check Complete.")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
