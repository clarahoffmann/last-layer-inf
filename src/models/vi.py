"""Variational inference for last-layer inference models."""
import torch
from typing import Tuple, Literal
from tqdm import tqdm
from torch.special import gammaln

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
                            L: int,
                            N: int) -> torch.Tensor:

    term1 = log_p_y_cond_w_tau_sq(Psi, mu_w, ys, tau_sq, L, N)
    term2 = log_p_w_cond_tau_sq(mu_w, tau_sq, L) 
    term3 = log_p_tau_sq(tau_sq, nu)

    return term1 + term2 + term3


#def get_log_likelihood_ridge():

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
    return - .5*torch.sum(w**2/tau_sq) - .5*L*torch.log(2*torch.pi) - 0.5*L*torch.log(tau_sq)

def log_p_tau_sq(tau_sq, nu):
    """Log-density of the scale-dependent prior of Klein, Kneib 2016"""
    return -.5*torch.log(tau_sq) + .5*torch.log(nu) - torch.sqrt(tau_sq/nu)

def log_p_sigma_eps_sq(sigma_eps_sq, a_sigma, b_sigma):
    log_prob = (
        a_sigma * torch.log(b_sigma)
        - gammaln(a_sigma)
        - (a_sigma + 1) * torch.log(sigma_eps_sq)
        - b_sigma / sigma_eps_sq
    )
    return log_prob


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


def fit_vi_post_hoc(ys: torch.tensor,
                    Psi: torch.tensor, 
                    lr: float, 
                    num_iter: float, 
                    method : Literal["closed_form", "ridge"],
                    sigma_eps_sq: float = 1.0,
                    sigma_0_sq: float = 1.0) -> Tuple[list[torch.tensor]]:

    if method == 'closed_form':
        lambdas, elbos = run_vi_closed_form(ys, Psi, num_iter, sigma_0_sq, sigma_eps_sq, lr)
    
    elif method == 'ridge':
        lambdas, elbos = run_vi_ridge(ys, Psi, num_iter, sigma_0_sq, sigma_eps_sq, lr)

    else:
        ValueError('Invalid method. Choose ridge or horseshoe as method')

    return lambdas, elbos


def predictive_posterior(Psi: torch.Tensor, mu: torch.Tensor, 
                         sigma: torch.Tensor, sigma_eps_sq: torch.Tensor):

    pred_mean = Psi @ mu
    pred_var = sigma_eps_sq + torch.sum((Psi ** 2) * (sigma ** 2).unsqueeze(0), dim=1)

    return pred_mean.detach().numpy(), pred_var.detach().numpy()
