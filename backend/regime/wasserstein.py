from scipy.stats import wasserstein_distance
import numpy as np

def compute_regime_distance(current_returns, previous_returns):
    """
    Computes Wasserstein distance between two return distributions.
    """
    return wasserstein_distance(current_returns, previous_returns)
