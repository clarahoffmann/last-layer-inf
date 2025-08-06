"""Utilities to compute coverage rates of prediction intervals."""

from scipy.stats import norm
import numpy as np
import torch

def get_coverage_gaussian(pred_mean, pred_std, y_true, levels):
    empirical_coverage = []

    for alpha in levels:
        z = norm.ppf(0.5 + alpha / 2.0)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std

        coverage = ((y_true >= lower) & (y_true <= upper)).mean()
        empirical_coverage.append(coverage)

    return np.array(empirical_coverage)



    
    
