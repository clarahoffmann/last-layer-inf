"""Variational inference for last-layer inference models."""
import math
import torch
from tqdm import tqdm
from typing import Tuple, Literal
from scipy.special import gammaln
from torch.optim import Adam



def get_h_params_ridge(ys: torch.Tensor, 
                            Psi: torch.Tensor,
                            mu_w: torch.Tensor,  
                            tau_sq: torch.Tensor,
                            sigma_eps_sq: torch.Tensor, 
                            nu: float,
                            a_sigma: float,
                            b_sigma: float,
                            L: int,
                            N: int) -> torch.Tensor:

    term1 = log_p_y_cond_w_sigma_sq(Psi, mu_w, ys, sigma_eps_sq, N)
    term2 = log_p_w_cond_tau_sq(mu_w, tau_sq, L) 
    term3 = log_p_tau_sq(tau_sq, nu)
    term4 = log_p_sigma_eps_sq(sigma_eps_sq, a_sigma, b_sigma)

    return term1 + term2 + term3 + term4

def log_p_y_cond_w_sigma_sq(Psi, mu_w, ys, sigma_eps_sq,  N):
    """Log of p(y | x, w, sigma_eps_sq) for the ridge prior (log-likelihood)."""
    pred_y = Psi @ mu_w
    log_likelihood = (
        - 0.5 * N * torch.log(sigma_eps_sq) 
        - 0.5 * torch.sum((ys - pred_y) ** 2) / sigma_eps_sq
    )
    
    return log_likelihood

def log_p_w_cond_tau_sq(w, tau_sq, L):
    """Log of p(w|tau^2) for the ridge prior."""
    return - .5*torch.sum(w**2/tau_sq) - .5*L*math.log(2*math.pi) - 0.5*L*torch.log(tau_sq)

def log_p_tau_sq(tau_sq, nu):
    """Log-density of the scale-dependent prior of Klein, Kneib 2016"""
    return -.5*torch.log(tau_sq) + .5*math.log(nu) - torch.sqrt(tau_sq/nu)

def log_p_sigma_eps_sq(sigma_eps_sq, a_sigma, b_sigma):
    log_prob1 = (a_sigma * math.log(b_sigma) - gammaln(a_sigma))
    log_prob2 = - (a_sigma + 1) * math.log(sigma_eps_sq + 1e-4) - b_sigma / sigma_eps_sq
    return log_prob1 +log_prob2


def run_vi_ridge(ys, Psi, num_iter, lr, S = 10):

    N, L = Psi.shape
    # we estimate the variational distribution for 
    # w_1, ..., w_L, log(tau_sq^2), log(sigma_eps^2)
    # -> we need dimension L + 2 for the variational distribution
    # for the ridge prior we have the parameters
    mu = torch.ones(L + 2, requires_grad=True)
    rho = torch.zeros(L + 2, requires_grad=True) 

    optimizer = torch.optim.Adam([mu, rho], lr=lr)
    elbos = []
    mus = []
    rhos = []

    for _ in tqdm(range(num_iter)):
        optimizer.zero_grad()


        # transformation to ensure sigma is pos.
        sigma = torch.exp(rho)
        sigma = torch.clamp(sigma, min=1e-6)
        
        # sample parameters
        eps = torch.randn(S, L + 2)
        params_sample = mu + sigma*eps

        # split parameters and transform log(tau^2) and log(sigma_eps^2) back
        w_samples = params_sample[:, :L]
        tau_sq_samples = torch.exp(params_sample[:,-2]) + 1e-5
        sigma_eps_sq_samples = torch.exp(params_sample[:,-1]) + 1e-5

        # posterior up to proportionality
        h_vals = []
        for s in range(S):
            h = get_h_params_ridge(
                ys=ys, Psi=Psi, 
                mu_w=w_samples[s],
                tau_sq=tau_sq_samples[s],
                sigma_eps_sq=sigma_eps_sq_samples[s],
                N=N, L=L, 
                a_sigma=2, b_sigma=2, nu=2.5
            )
            h_vals.append(h)
        h_params = torch.stack(h_vals).mean()
        
        # need to subtract logprob of gaussian variational distribution here (diag. gaussian with mean
        # mu_w and variance sigma_w^2).
        log_q_params = (- 0.5 * (L+2) * math.log(2 * math.pi) \
                        - torch.sum(torch.log(sigma), dim=-1) \
                        - 0.5 * torch.sum((((params_sample - mu) / sigma) ** 2), dim = -1)).mean()
        
        elbo = h_params + log_q_params
        loss = - elbo
        elbos.append(elbo.item())
        mus.append(mu.detach())
        rhos.append(rho.detach())

        loss.backward()
        optimizer.step()

    lambdas = {
            'mu': mu.detach(),
            'sigma': torch.exp(rho.detach()),
        }


    return lambdas, elbos, mus, rhos

def fit_vi_post_hoc(ys: torch.tensor,
                    Psi: torch.tensor, 
                    num_iter: float, 
                    method : Literal["ridge"],
                    mu_0: torch.Tensor,
                    sigma_eps_sq: float = 1.0,
                    sigma_0_sq: float = 1.0,
                    lr: float = 1e-4, 
                    ) -> Tuple[list[torch.tensor]]:

    if method == 'ridge':
        lambdas, elbos, mus, rhos = run_vi_ridge(ys = ys, Psi = Psi,
                                      num_iter = num_iter, lr = lr)
        return lambdas, elbos,  mus, rhos

    else:
        ValueError('Invalid method. Choose ridge or horseshoe as method')

    


def predictive_posterior(Psi: torch.Tensor, mu: torch.Tensor, 
                         Sigma_w: torch.Tensor, sigma_eps_sq: torch.Tensor):

    pred_mean = Psi @ mu
    pred_var = sigma_eps_sq + (Psi @ Sigma_w @ Psi.T)

    return pred_mean.detach().numpy(), pred_var.detach().numpy()

def run_last_layer_vi_closed_form(model, ys_train, Psi, sigma_eps_sq, lr = 1e-2, temperature = 1, num_epochs = 1000):

    optimizer_vi = Adam(model.parameters(), lr=lr)

    elbos = []
    for epoch in range(num_epochs):
        optimizer_vi.zero_grad()
        
        # compute y_hat with current var. parameters
        pred_y_mu= model.forward_det(Psi)
        
        # get current covariance of var. distribution
        L = model.get_L().squeeze() # Cholesky factor
        Sigma_w = L @ L.T # actual covariance matrix

        # get kl term
        kl = model.kl_divergence()

        # get likelihood term
        log_likelihood = (
        -0.5 * Psi.shape[0] * math.log(2 * torch.pi * (sigma_eps_sq / temperature))
        - 0.5 * temperature / sigma_eps_sq * torch.sum((ys_train - pred_y_mu) ** 2)
        - 0.5 * temperature / sigma_eps_sq * torch.diagonal(Psi @ Sigma_w @ Psi.T).sum())
        
        # combine for final ELBO
        elbo = (
            log_likelihood
            - kl
        )
        loss = -elbo
        loss.backward()
        optimizer_vi.step()
        if epoch % 100 == 0:
            print(f"VI epoch {epoch} ELBO: {elbo.item():.3f} \n 'log likelihood: {log_likelihood.item():.3f}")

        elbos.append(elbo.item())
    
    return model, elbos
