# -*- coding: utf-8 -*-
# English Comments
import pandas as pd
import numpy as np
import os
import argparse
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import dimension calculation functions from our new module
from metrics import dimensions


# This is the main entry point for running the entire IRIS benchmark evaluation.
# It orchestrates the calculation of metrics, score aggregation, and visualization.

class EvalConfig:
    """Configuration class for the main evaluation pipeline."""
    # --- Directories ---
    # This structure assumes preprocessing scripts have been run
    PROCESSED_DATA_DIR = "processed_data"
    ANNOTATIONS_DIR = "data/annotations"
    OUTPUT_DIR = "results/final_evaluation"

    # --- Filename conventions within processed directories ---
    GEN_CLASSIFICATION_FILENAME = "final_classified_report_with_profession.csv"
    BIS_GEN_UNIFIED_REPORT_PATTERN = "unified_report_{model_name}.csv"
    UND_VQA_FILENAME = "mapped_vqa_results.csv"  # From preprocessing
    RFS_UND_JSD_INPUT_FILE = "tournament_probe_wins.csv"  # Assumed to be in processed_data/vqa/{model_name}
    BIS_UND_RESULTS_PATTERN = "*_individual_results.csv"

    # --- Annotation filenames ---
    ANNOTATIONS_FILENAME = "annotations.csv"
    REAL_DATA_US_CSV = "real_world_stats_us.csv"
    REAL_DATA_EU_CSV = "real_world_stats_eu.csv"

    # --- Final Score Parameters (from new_total.py) ---
    TOTAL_SCORE_SCALING_FACTOR_S = 50000
    TOTAL_SCORE_DECAY_CONSTANT_K = 5


def normalize_deviation_space(df: pd.DataFrame) -> pd.DataFrame:
    """Applies normalization to transform all metrics into a comparable deviation space."""
    print("\n>>> Normalizing the space using theoretical & practical bounds...")
    df_normalized = df.copy()
    for col in df_normalized.columns:
        # Metrics already in [0, 1] or treated as direct deviation
        if col.startswith(('RD_', 'JSD_', 'AD_', 'SPD_', 'dhr_', 'Penalty_ΔGSR')):
            continue
        # AC_diff is normalized by its theoretical max range (9, i.e., 10 - 1)
        elif col.startswith('ac_diff_'):
            df_normalized[col] = df_normalized[col] / 9.0
        # Unbounded penalties and SDS are log-transformed
        elif col.startswith(('Penalty_', 'AbsSDS_')):
            df_normalized[col] = np.log1p(df_normalized[col])
    print("Normalization complete.")
    return df_normalized


def visualize_space(df_normalized: pd.DataFrame, output_dir: str):
    """Generates and saves PCA and t-SNE plots of the fairness space."""
    if df_normalized.empty or len(df_normalized) < 2:
        print("Warning: Visualization requires at least 2 models. Skipping.")
        return

    print("\n>>> Visualizing the high-dimensional space...")
    models = df_normalized.index

    # Use Z-Score scaling for visualization to maximize variance between models
    visual_data = StandardScaler().fit_transform(df_normalized)

    # PCA Visualization
    try:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(visual_data)
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=models)

        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue=df_pca.index, s=200, palette='viridis', legend='full')
        for i, model in enumerate(df_pca.index):
            plt.text(df_pca.iloc[i]['PC1'] + 0.05, df_pca.iloc[i]['PC2'], model, fontsize=9)
        plt.title('Model Fairness Map (PCA Projection)', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.grid(True);
        plt.axhline(0, c='grey', lw=1, ls='--');
        plt.axvline(0, c='grey', lw=1, ls='--')
        plt.savefig(os.path.join(output_dir, "fairness_map_pca.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to generate PCA plot: {e}")

    # t-SNE Visualization
    try:
        # Perplexity must be less than the number of samples
        perplexity = max(1, min(30, len(df_normalized) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        tsne_results = tsne.fit_transform(visual_data)
        df_tsne = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'], index=models)

        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='TSNE1', y='TSNE2', data=df_tsne, hue=df_tsne.index, s=200, palette='viridis', legend='full')
        for i, model in enumerate(df_tsne.index):
            plt.text(df_tsne.iloc[i]['TSNE1'] + 0.05, df_tsne.iloc[i]['TSNE2'], model, fontsize=9)
        plt.title(f'Model Fairness Map (t-SNE Projection, Perplexity={perplexity})', fontsize=16)
        plt.xlabel('t-SNE Dimension 1');
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "fairness_map_tsne.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to generate t-SNE plot: {e}")

    print("Visualizations saved.")


def main():
    config = EvalConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 50);
    print("      IRIS Benchmark Evaluation Pipeline      ");
    print("=" * 50)

    # --- Step 1: Calculate raw sub-metrics for all 6 dimensions ---
    df_ifs_gen = dimensions.calculate_ifs_gen_metrics(os.path.join(config.PROCESSED_DATA_DIR, "generation"), config)
    df_rfs_gen = dimensions.calculate_rfs_gen_metrics(os.path.join(config.PROCESSED_DATA_DIR, "generation"), config)
    df_bis_gen = dimensions.calculate_bis_gen_metrics(os.path.join(config.PROCESSED_DATA_DIR, "bis_gen"), config)
    df_ifs_und = dimensions.calculate_ifs_und_metrics(os.path.join(config.PROCESSED_DATA_DIR, "vqa"), config)
    df_rfs_und = dimensions.calculate_rfs_und_metrics(os.path.join(config.PROCESSED_DATA_DIR, "vqa"), config)
    df_bis_und = dimensions.calculate_bis_und_metrics(os.path.join(config.PROCESSED_DATA_DIR, "bis_und"), config)

    # --- Step 2: Build and Normalize the Total Deviation Space ---
    print("\n>>> Merging all sub-metrics to build Total Deviation Space...")
    all_dfs = [df for df in [df_ifs_gen, df_rfs_gen, df_bis_gen, df_ifs_und, df_rfs_und, df_bis_und] if
               df is not None and not df.empty]
    if not all_dfs:
        print("Error: No metrics were calculated. Halting.");
        return

    df_total_raw = reduce(lambda left, right: pd.merge(left, right, on='model', how='outer'), all_dfs).fillna(0)
    df_total_raw.set_index('model', inplace=True)

    df_total_normalized = normalize_deviation_space(df_total_raw)

    output_path = os.path.join(config.OUTPUT_DIR, "total_deviation_space_normalized.csv")
    df_total_normalized.to_csv(output_path)
    print(f"Normalized Deviation Space matrix saved to: {output_path}")

    # --- Step 3: Calculate Final Scores ---
    print("\n>>> Calculating final IRIS scores...")
    distances = np.linalg.norm(df_total_normalized.values, axis=1)
    final_scores = config.TOTAL_SCORE_SCALING_FACTOR_S * np.exp(-config.TOTAL_SCORE_DECAY_CONSTANT_K * distances)

    df_scores = pd.DataFrame({
        'model': df_total_normalized.index,
        'Total_Deviation_Distance': distances,
        'IRIS_Total_Score': final_scores
    }).sort_values('IRIS_Total_Score', ascending=False)

    print("\n--- Final Model Ranking ---");
    print(df_scores.to_string(index=False))
    scores_path = os.path.join(config.OUTPUT_DIR, "final_iris_scores.csv")
    df_scores.to_csv(scores_path, index=False)
    print(f"Final scores saved to: {scores_path}")

    # --- Step 4: Visualize ---
    visualize_space(df_total_normalized, config.OUTPUT_DIR)

    print("\n✅ Evaluation pipeline completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full IRIS benchmark evaluation pipeline.")
    # Example argument: you can add a way to run on a subset of models
    parser.add_argument('--models', nargs='+',
                        help="Optional: A list of specific model names to run the evaluation on.")
    args = parser.parse_args()
    main()

