"""Gibbs sampler for last-layer inference with a ridge prior.

# Three steps to get the predictive mean and variance 
# 1. sample density over a grid of y values
# 2. Sample 1,000 y[s] based on the density hat{p}(y | x)
# 3. Compute mean and variance of the samples. """

import torch
import numpy as np
from tqdm import tqdm
from torch.distributions import MultivariateNormal, InverseGamma, Normal

def w_cond_D_tau_sq_sigma_sq(Psi, ys, tau_sq, sigma_eps_sq):
    d = Psi.shape[1]
    Sigma_N_inv = (1/tau_sq)*torch.eye(d) + (1/sigma_eps_sq)*(Psi.T @ Psi)
    Sigma_N = torch.linalg.inv(Sigma_N_inv).detach()
    mu_N = Sigma_N @ ((1/sigma_eps_sq)*(Psi.T @ ys)).detach().squeeze()

    dist = MultivariateNormal(loc = mu_N, covariance_matrix = Sigma_N)

    return dist.rsample()

def tau_sq_cond_D_w_sigma_sq(w, a_tau, b_tau, L):
    concentration = a_tau + .5*L
    rate = b_tau + .5*(w.pow(2).sum().sqrt()) # only with this the results look good, what is wrong?
    dist = InverseGamma(concentration = concentration , 
                        rate = rate)
    return dist.sample()


def sigma_sq_cond_D_tau_sq_w(Psi, ys, w, a_sigma, b_sigma, N):
    concentration = a_sigma + .5*N
    rate = b_sigma + .5*((ys - Psi @ w).pow(2).sum().sqrt()) 
    dist = InverseGamma(concentration = concentration, 
                        rate = rate)
    return dist.sample()


def gibbs_sampler(Psi: torch.tensor, ys: torch.tensor, 
                  a_tau: float, b_tau: float, 
                  a_sigma: float, b_sigma: float, num_iter: int, 
                  warm_up: int):

    N, L = Psi.shape

    samples = {
        'w': [],
        'tau_sq': [],
        'sigma_sq': []
    }

    # Initialize parameters
    tau_sq = .2
    sigma_sq = .2
    with torch.no_grad():
        for i in tqdm(range(num_iter)):
            
            # draw w | D, \tau^2, \sigma_eps^2 (Gaussian)
            w = w_cond_D_tau_sq_sigma_sq(Psi, ys, tau_sq, sigma_sq)
            samples['w'].append(w)

            # draw \tau^2 | D, w,  \sigma_eps^2 (inverse-gamma)
            tau_sq = tau_sq_cond_D_w_sigma_sq(w, a_tau, b_tau, L)
            samples['tau_sq'].append(tau_sq)

            #  \sigma_eps^2 | D, \tau^2, w (inverse-gamma)
            sigma_sq = sigma_sq_cond_D_tau_sq_w(Psi, ys, w, a_sigma, b_sigma, N)
            samples['sigma_sq'].append(sigma_sq)
        
    return samples['w'][warm_up:], samples['tau_sq'][warm_up:], samples['sigma_sq'][warm_up:]


def py_cond_x_w_sigma_eps(y, psi, w, sigma_eps_sq):
    dist = Normal(loc = psi @ w, scale = torch.sqrt(sigma_eps_sq))
    return dist.log_prob(y)

def get_pred_post_dist(psi, w_sample, sigma_sq_sample, ys_grid):
    """Computes the posterior preditive distribution (pdf), pred. mean
     and pred. standard deviation based on Monte-Carlo samples of the 
     posterior weight and post. obs. noise. The pdf is evaluated at ys_grid.
     The pred. mean and standard deviation are integrated numericall via by \int y *pdf(y) dy
     and 
    """
    p_hats = torch.exp(torch.stack([py_cond_x_w_sigma_eps(ys_grid, 
                                                                     psi , 
                                                                     w_sample[i], 
                                                                     sigma_sq_sample[i]) for i in range(len(w_sample))]))
                                                                    

    p_hats = torch.mean(p_hats, dim = 0)                              
    y_mean = torch.trapz(p_hats*ys_grid,ys_grid)

    # Compute variance: E[x^2] - (E[x])^2
    mean_sq_i = torch.trapz(p_hats * ys_grid**2, ys_grid)
    y_var = mean_sq_i - y_mean**2

    return p_hats.detach().numpy(), y_mean.detach().numpy(), y_var.detach().numpy()

def get_prediction_interval_coverage(ys_grid, ys, p_hats, levels):

    empirical_coverage = []
    
    for level in tqdm(levels):
        # Normalize p_hats in case it doesn't integrate to 1
        area = torch.trapz(p_hats, ys_grid)
        p_normalized = p_hats / area

        # Compute cumulative distribution (CDF)
        cdf = torch.cumsum(p_normalized, dim=0)
        cdf = cdf / cdf[-1]  # Normalize CDF

        # Define lower and upper tail mass
        lower_tail = (1 - level) / 2
        upper_tail = 1 - lower_tail

        # Interpolate to find quantiles
        lower_idx = torch.searchsorted(cdf, lower_tail)
        upper_idx = torch.searchsorted(cdf, upper_tail)

        # Clamp indices to stay in bounds
        lower_idx = min(lower_idx, len(ys_grid) - 1)
        upper_idx = min(upper_idx, len(ys_grid) - 1)

        lower_bound = ys_grid[lower_idx]
        upper_bound = ys_grid[upper_idx]

        covered = ((ys >= lower_bound) & (ys <= upper_bound)).float().mean().item()

        empirical_coverage.append(covered)
    
    return np.array(empirical_coverage)


