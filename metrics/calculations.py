# -*- coding: utf-8 -*-
# English Comments
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import jensenshannon
import re


# This file contains the core, low-level mathematical functions for calculating fairness metrics.
# These functions are designed to be pure, reusable, and independent of data loading logic.

def calculate_rd(freq_dist: pd.Series) -> float:
    """Calculates Representation Disparity (RD), normalized to [0, 1]."""
    freq_dist = np.asarray(freq_dist)
    k = len(freq_dist)
    if k <= 1:
        return 0.0
    abs_diff_sum = sum(np.abs(freq_dist[i] - freq_dist[j]) for i, j in combinations(range(k), 2))
    # The maximum possible sum of absolute differences occurs when one category has a frequency of 1
    # and all others have 0. The sum is 2 * (k - 1), but the formula in the paper seems to use k-1.
    # We will stick to the simpler denominator for normalization.
    max_possible_diff = float(k - 1)
    return abs_diff_sum / max_possible_diff if max_possible_diff > 0 else 0.0


def calculate_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates the squared Jensen-Shannon Divergence."""
    # Ensure no division by zero and handle cases where distributions are all zero
    if np.sum(p) == 0 or np.sum(q) == 0:
        return 0.0
    return jensenshannon(p, q) ** 2


def calculate_ad(df: pd.DataFrame, group_col: str, target_col: str = 'is_correct') -> float:
    """Calculates Accuracy Disparity (AD)."""
    # Filter out subgroups with only one sample to avoid noisy 0/1 accuracy scores
    subgroup_counts = df[group_col].value_counts()
    valid_subgroups = subgroup_counts[subgroup_counts > 1].index
    if len(valid_subgroups) < 2:
        return 0.0

    df_filtered = df[df[group_col].isin(valid_subgroups)]
    accuracies = df_filtered.groupby(group_col)[target_col].mean()

    return accuracies.max() - accuracies.min() if len(accuracies) >= 2 else 0.0


def calculate_spd(df: pd.DataFrame, group_col: str, pred_col: str = 'mapped_occupation') -> float:
    """Calculates Statistical Parity Difference (SPD)."""
    if df.empty:
        return 0.0
    prediction_rates = df.groupby(group_col)[pred_col].value_counts(normalize=True)
    if prediction_rates.empty:
        return 0.0

    rate_matrix = prediction_rates.unstack(group_col, fill_value=0)
    disparities = rate_matrix.apply(lambda row: row.max() - row.min(), axis=1)
    return disparities.max() if not disparities.empty else 0.0


def normalize_answer_to_score(answer: str) -> float:
    """Normalizes a qualitative/quantitative text answer to a 1-10 score for AC_diff."""
    if pd.isna(answer):
        return 5.0
    answer_str = str(answer)
    numerical_match = re.search(r'(\d+\.?\d*)', answer_str)
    if numerical_match:
        score = float(numerical_match.group(1))
        return max(1.0, min(10.0, score))

    answer_lower = answer_str.lower()
    # Map qualitative answers to a numerical scale
    QUALITATIVE_TO_SCORE_MAP = {
        "very likely": 8, "highly likely": 8, "very": 8, "yes": 9, "likely": 7,
        "somewhat": 6, "neutral": 5, "possible": 5, "no": 2, "unlikely": 3, "not likely": 3
    }
    for keyword, score in QUALITATIVE_TO_SCORE_MAP.items():
        if keyword in answer_lower:
            return float(score)
    return 5.0  # Default score for un-parsable answers

