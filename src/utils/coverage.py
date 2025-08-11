"""Utilities to compute coverage rates of prediction intervals."""

from scipy.stats import norm
import numpy as np
import torch

def get_coverage_gaussian(pred_mean, pred_std, y_true, levels):
    empirical_coverage = []

    for level in levels:
        alpha = 1 - level

        lower = pred_mean + pred_std * norm.ppf(alpha / 2)
        upper = pred_mean + pred_std * norm.ppf(1 - alpha / 2)

        coverage = ((y_true >= lower) & (y_true <= upper)).mean()
        empirical_coverage.append(coverage)

    return np.array(empirical_coverage)

def get_coverage_y_hats(y_samples, y_true, levels):
    empirical_coverage = [] 

    for level in levels:
        lower = (1 - level) / 2
        upper = 1 - lower

        lower = torch.quantile(y_samples, lower, dim=1)
        upper = torch.quantile(y_samples, upper, dim=1)
        coverage = ((y_true >= lower) & (y_true <= upper)).float().mean()
        empirical_coverage.append(coverage)
    
    return np.array(empirical_coverage)
    
    
