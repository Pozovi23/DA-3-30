import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.datasets import load_iris, make_regression

from calculate_spearman_correlation import calculate_spearman_correlation


def load_dataset(
    dataset_type: str = "synthetic",
    n_samples: int = 100,
    noise: float = 5.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load dataset (synthetic with nonlinear relationship or Iris).

    Parameters
    ----------
    dataset_type : str, optional
        Type of dataset ("synthetic" or "iris"), by default "synthetic"
    n_samples : int, optional
        Number of samples for synthetic dataset, by default 100
    noise : float, optional
        Amount of noise for synthetic dataset, by default 5.0
    random_state : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features (X) and target (y) as numpy arrays

    Raises
    ------
    ValueError
        If dataset_type is not "synthetic" or "iris"
    """
    try:
        if dataset_type == "synthetic":
            # Generate synthetic data with linear relationship
            X, y = make_regression(
                n_samples=n_samples,
                n_features=1,
                noise=noise,
                random_state=random_state,
            )

            # Normalize y to prevent overflow in exponential transformation
            y = (y - np.mean(y)) / np.std(y)

            # Create nonlinear relationship
            y_nonlinear = np.exp(2 * y)

            return X.ravel(), y_nonlinear

        elif dataset_type == "iris":
            iris = load_iris()
            X = iris.data[:, 0]  # Sepal length
            y = iris.data[:, 2]  # Petal length

            # Normalize both features for consistency
            X = (X - np.mean(X)) / np.std(X)
            y = (y - np.mean(y)) / np.std(y)

            return X, y

        else:
            raise ValueError("Invalid dataset_type. Use 'synthetic' or 'iris'.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return np.array([]), np.array([])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Pearson correlation between two variables.

    Parameters
    ----------
    x : np.ndarray
        First feature array
    y : np.ndarray
        Second feature array

    Returns
    -------
    float
        Pearson correlation coefficient

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty
    """
    try:
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("Input arrays must have same length and cannot be empty")

        pearson_corr, _ = stats.pearsonr(x, y)

        return pearson_corr
    except Exception as e:
        print(f"Error calculating Pearson correlation: {e}")
        return np.nan


def analyze_correlation_difference(spearman_corr: float, pearson_corr: float) -> str:
    """
    Analyze difference between Spearman and Pearson correlations.

    Parameters
    ----------
    spearman_corr : float
        Spearman correlation coefficient
    pearson_corr : float
        Pearson correlation coefficient

    Returns
    -------
    str
        Explanation of the difference between correlations

    Raises
    ------
    ValueError
        If correlation coefficients are NaN
    """
    try:
        if np.isnan(spearman_corr) or np.isnan(pearson_corr):
            raise ValueError("Correlation coefficients cannot be NaN")

        diff = abs(spearman_corr - pearson_corr)

        explanation = (
            f"Spearman correlation: {spearman_corr:.3f}\n"
            f"Pearson correlation: {pearson_corr:.3f}\n"
            f"Difference: {diff:.3f}\n\n"
        )

        return explanation
    except Exception as e:
        return f"Error analyzing correlations: {e}"


def visualize_data(
    x: np.ndarray, y: np.ndarray, dataset_type: str = "synthetic", save_path: str = None
) -> None:
    """
    Create a scatter plot of the relationship between two variables.

    Parameters
    ----------
    x : np.ndarray
        First feature array
    y : np.ndarray
        Second feature array
    dataset_type : str, optional
        Type of dataset ("synthetic" or "iris"), by default "synthetic"
    save_path : str, optional
        Path to save the plot, by default None
    """
    try:
        title = (
            "Very Strong Nonlinear (Exponential) Relationship"
            if dataset_type == "synthetic"
            else "Iris: Sepal Length vs. Petal Length"
        )
        ylabel = (
            "Feature Y (Enhanced Exponential)"
            if dataset_type == "synthetic"
            else "Petal Length (Normalized)"
        )

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.5)
        plt.title(title)
        plt.xlabel(
            "Feature X" if dataset_type == "synthetic" else "Sepal Length (Normalized)"
        )
        plt.ylabel(ylabel)
        plt.grid(True)

        if save_path:
            (
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.dirname(save_path)
                else None
            )
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.close()

    except Exception as e:
        print(f"Error in visualization: {e}")


def main(dataset_type: str = "synthetic", save_path: str = None) -> None:
    """
    Main function to execute correlation analysis.

    Parameters
    ----------
    dataset_type : str, optional
        Type of dataset ("synthetic" or "iris"), by default "synthetic"
    save_path : str, optional
        Path to save the plot, by default None
    """
    try:
        X, y = load_dataset(dataset_type=dataset_type)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Failed to load dataset")

        visualize_data(X, y, dataset_type, save_path)

        spearman_corr = calculate_spearman_correlation(X, y)
        pearson_corr = calculate_pearson_correlation(X, y)

        result = analyze_correlation_difference(spearman_corr, pearson_corr)
        print(result)

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    synthetic_save_path = "output/synthetic_plot.png"
    iris_save_path = "output/iris_plot.png"

    print("Running with synthetic dataset:")
    main(dataset_type="synthetic", save_path=synthetic_save_path)

    print("\nRunning with Iris dataset:")
    main(dataset_type="iris", save_path=iris_save_path)