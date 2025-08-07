"""Variational inference for last-layer inference models."""
import math
import torch
from tqdm import tqdm
from typing import Tuple, Literal
from scipy.special import gammaln

def get_log_likelihood_closed_form(ys: torch.Tensor, 
                                   Psi: torch.Tensor,
                                   mu_w: torch.Tensor,  
                                   sigma_eps_sq: torch.Tensor, 
                                   sigma_w_sq: torch.Tensor, 
                                   N: int) -> torch.Tensor:
    pred_y = Psi @ mu_w
    log_likelihood = (
        - 0.5 * N * torch.log(sigma_eps_sq) 
        - 0.5 * torch.sum((ys - pred_y) ** 2) / sigma_eps_sq
    )

    var_component = 0.5 * (1.0 / sigma_eps_sq) * torch.sum((Psi ** 2) / sigma_w_sq, dim=1)

    likelihood = torch.sum(log_likelihood) - torch.sum(var_component)
    return likelihood



def get_log_likelihood_ridge(ys: torch.Tensor, 
                            Psi: torch.Tensor,
                            mu_w: torch.Tensor,  
                            tau_sq: torch.Tensor,
                            sigma_eps_sq: torch.Tensor, 
                            nu: float,
                            a_sigma: float,
                            b_sigma: float,
                            L: int,
                            N: int) -> torch.Tensor:

    term1 = log_p_y_cond_w_tau_sq(Psi, mu_w, ys, tau_sq, L, N)
    term2 = log_p_w_cond_tau_sq(mu_w, tau_sq, L) 
    term3 = log_p_tau_sq(tau_sq, nu)
    term4 = log_p_sigma_eps_sq(sigma_eps_sq, a_sigma, b_sigma)

    return term1 + term2 + term3 + term4

def log_p_y_cond_w_tau_sq(Psi, mu_w, ys, sigma_eps_sq, L, N):
    """Log of p(y | x, w, tau_sq) for the ridge prior (log-likelihood)."""
    pred_y = Psi @ mu_w
    log_likelihood = (
        - 0.5 * N * torch.log(sigma_eps_sq) 
        - 0.5 * torch.sum((ys - pred_y) ** 2) / sigma_eps_sq
    )
    
    return log_likelihood

def log_p_w_cond_tau_sq(w, tau_sq, L):
    """Log of p(w|tau^2) for the ridge prior."""
    return - .5*torch.sum(w**2/tau_sq) - .5*L*math.log(2*torch.pi) - 0.5*L*torch.log(tau_sq)

def log_p_tau_sq(tau_sq, nu):
    """Log-density of the scale-dependent prior of Klein, Kneib 2016"""
    return -.5*torch.log(tau_sq) + .5*math.log(nu) - torch.sqrt(tau_sq/nu)

def log_p_sigma_eps_sq(sigma_eps_sq, a_sigma, b_sigma):
    log_prob1 = (a_sigma * math.log(b_sigma) - gammaln(a_sigma))
    log_prob2 = - (a_sigma + 1) * math.log(sigma_eps_sq + 1e-4) - b_sigma / sigma_eps_sq
    return log_prob1 +log_prob2


#def get_log_likelihood_horseshoe():

def run_vi_ridge(ys, Psi, num_iter, sigma_w, mu_w, sigma_eps_sq, sigma_0_sq, N, lr):

    N, L = Psi.shape
    mu = torch.zeros(L, requires_grad=True)
    rho = torch.zeros(L, requires_grad=True) 

    optimizer = torch.optim.Adam([mu, rho], lr=lr)
    elbos = []

    for _ in range(num_iter):
        optimizer.zero_grad()

        log_likelihood = get_log_likelihood_ridge(ys, Psi, mu_w, sigma_eps_sq, sigma_w**2, N)
        kld = 0.5 * torch.sum(
            torch.log(sigma_w**2 / sigma_0_sq) + (sigma_0_sq**-1) * (mu**2 + sigma_w**2) - 1
        )
        elbo = log_likelihood + kld
        loss = - elbo

        loss.backward()
        optimizer.step()

    lambdas = {
            'mu': mu.detach(),
            'tau_sq': torch.nn.functional.softplus(rho.detach()),
            'sigma_eps': torch.nn.functional.softplus(rho.detach())
        }


    return lambdas, elbos


def run_vi_closed_form(ys, Psi, num_iter, sigma_0_sq, sigma_eps_sq, lr):

    N, L = Psi.shape
    mu = torch.zeros(L, requires_grad=True)
    rho = torch.zeros(L, requires_grad=True) 

    optimizer = torch.optim.Adam([mu, rho], lr=lr)
    elbos = []
    
    for _ in tqdm(range(num_iter)):
        optimizer.zero_grad()

        # transformation to ensure sigma is pos.
        sigma_w = torch.nn.functional.softplus(rho)
        sigma_w = torch.clamp(sigma_w, min=1e-4)
        
        # compute elbo in closed form
        log_likelihood = get_log_likelihood_closed_form(ys, Psi, mu, sigma_eps_sq, sigma_w**2, N)
        kld = 0.5 * torch.sum(
            torch.log(sigma_w**2 / sigma_0_sq) + (sigma_0_sq**-1) * (mu**2 + sigma_w**2) - 1
        )

        elbo = log_likelihood - kld
        loss = - elbo

        loss.backward()
        optimizer.step()

        elbos.append(elbo.item())
    
    lambdas = {
            'mu': mu.detach(),
            'sigma': torch.nn.functional.softplus(rho.detach()),
        }

    return lambdas, elbos


def run_vi_ridge(ys, Psi, num_iter, sigma_eps_sq, sigma_0_sq, lr):

    N, L = Psi.shape
    # we estimate the variational distribution for 
    # w_1, ..., w_L, tau_sq, sigma_eps
    # -> we need dimension L + 2 for the variational distribution
    mu = torch.zeros(L + 2, requires_grad=True)
    rho = torch.zeros(L + 2, requires_grad=True) 

    optimizer = torch.optim.Adam([mu, rho], lr=lr)
    elbos = []

    for _ in tqdm(range(num_iter)):
        optimizer.zero_grad()

        # transformation to ensure sigma is pos.
        sigma = torch.nn.functional.softplus(rho)
        sigma = torch.clamp(sigma, min=1e-4)
        
        # split mu and sigmas
        mu_w = mu[:L]
        mu_tau_sq = torch.clamp(torch.nn.functional.softplus(mu[-2]), min=1e-4)
        mu_sigma_eps = torch.clamp(torch.nn.functional.softplus(mu[-1]), min=1e-4)

        log_likelihood = get_log_likelihood_ridge(ys = ys, Psi =Psi, 
                                                  mu_w = mu_w, 
                                                  tau_sq = mu_tau_sq**2,
                                                  sigma_eps_sq = mu_sigma_eps, 
                                                  N =  N, L = L, a_sigma = 0.2, b_sigma = 0.2, nu = 3)
        
        # need a different kld here
        kld = 0
        elbo = log_likelihood + kld
        loss = - elbo
        elbos.append(elbo.item())

        loss.backward()
        optimizer.step()

    lambdas = {
            'mu': mu.detach(),
            'sigma': torch.nn.functional.softplus(rho.detach()),
        }


    return lambdas, elbos

def fit_vi_post_hoc(ys: torch.tensor,
                    Psi: torch.tensor, 
                    num_iter: float, 
                    method : Literal["closed_form", "ridge"],
                    sigma_eps_sq: float = 1.0,
                    sigma_0_sq: float = 1.0,
                    lr: float = 1e-4) -> Tuple[list[torch.tensor]]:

    if method == 'closed_form':
        lambdas, elbos = run_vi_closed_form(ys, Psi, num_iter, sigma_0_sq, sigma_eps_sq, lr)
    
    elif method == 'ridge':
        lambdas, elbos = run_vi_ridge(ys = ys, Psi = Psi, sigma_0_sq = 0, 
                                      num_iter = num_iter, sigma_eps_sq = sigma_eps_sq, lr = lr)

    else:
        ValueError('Invalid method. Choose ridge or horseshoe as method')

    return lambdas, elbos


def predictive_posterior(Psi: torch.Tensor, mu: torch.Tensor, 
                         sigma: torch.Tensor, sigma_eps_sq: torch.Tensor):

    pred_mean = Psi @ mu
    pred_var = sigma_eps_sq + torch.sum((Psi ** 2) * (sigma ** 2).unsqueeze(0), dim=1)

    return pred_mean.detach().numpy(), pred_var.detach().numpy()
