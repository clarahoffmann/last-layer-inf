"""Variational inference for last-layer inference models."""
import torch
from typing import Tuple
from tqdm import tqdm

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

    var_component = 0.5 * (1.0 / sigma_eps_sq) * torch.sum((Psi ** 2) / sigma_w_sq, dim=1)  # (n,)

    likelihood = torch.sum(log_likelihood) - torch.sum(var_component)
    return likelihood


def fit_vi_post_hoc(ys: torch.tensor,
                    Psi: torch.tensor, 
                    lr: float, 
                    num_iter: float, 
                    sigma_eps_sq: float = 1.0,
                    sigma_p_sq: float = 1.0) -> Tuple[list[torch.tensor]]:

    N, L = Psi.shape
    mu = torch.zeros(L, requires_grad=True)
    rho = torch.zeros(L, requires_grad=True) 

    optimizer = torch.optim.Adam([mu, rho], lr=lr)
    elbos = []
    
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()

        # transformation to ensure sigma is pos.
        sigma_w = torch.nn.functional.softplus(rho)
        
        # compute elbo in closed form
        log_likelihood = get_log_likelihood_closed_form(ys, Psi, mu, sigma_eps_sq, sigma_w**2, N)
        kld = 0.5 * torch.sum(
            torch.log(sigma_w**2 / sigma_p_sq) + (sigma_p_sq**-1) * (mu**2 + sigma_w**2) - 1
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

def predictive_posterior(Psi: torch.Tensor, mu: torch.Tensor, 
                         sigma: torch.Tensor, sigma_eps_sq: torch.Tensor):

    pred_mean = Psi @ mu
    pred_var = sigma_eps_sq + torch.sum((Psi ** 2) * (sigma ** 2).unsqueeze(0), dim=1)

    return pred_mean.detach().numpy(), pred_var.detach().numpy()
