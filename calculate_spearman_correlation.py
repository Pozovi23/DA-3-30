import numpy as np
import scipy.stats as stats


def calculate_spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Spearman correlation between two variables.

    Args:
        x: First feature array
        y: Second feature array

    Returns:
        float: Spearman correlation coefficient
    """
    try:
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("Input arrays must have same length and cannot be empty")

        spearman_corr, _ = stats.spearmanr(x, y)

        return spearman_corr
    except Exception as e:
        print(f"Error calculating Spearman correlation: {e}")
        return np.nan
