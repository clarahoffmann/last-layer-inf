import torch


"""Hamiltonian Monte Carlo (HMC) for the reverse conditional."""

from typing import Callable

import torch
import math
from torch import nn
from tqdm import tqdm


# pylint: disable = C0116
class HMC(nn.Module):
    """Hamiltonian Monte Carlo implementation
    based on https://github.com/GavinPHR/HMC-VAE.
    """

    def __init__(
        self,
        dim: int,
        T: int,
        L: int,
        step_size: torch.Tensor = torch.tensor(0.2),
    ):
        super().__init__()
        self.dim = dim
        self.log_prob: Callable = None  # type: ignore
        self.T = T
        self.L = L
        self.step_size = step_size

    def register_log_prob(self, log_prob: Callable):
        self.log_prob = log_prob

    def grad_log_prob(self, x):
        with torch.enable_grad():
            x = x.clone().detach()
            x.requires_grad = True
            logprob = self.log_prob(x) #.sum()
            grad = torch.autograd.grad(logprob, x)[0]
            # clamp gradients to avoid exploding
            # epsilon values
            #grad = torch.clamp(grad, min=-100, max=100)
            return grad

    def leapfrog(self, x, p):
        eps = self.step_size
        p = p + 0.5 * eps * self.grad_log_prob(x)
        for _ in range(self.L - 1):
            x = x + eps * p
            p = p + eps * self.grad_log_prob(x)
        x = x + eps * p
        p = p + 0.5 * eps * self.grad_log_prob(x)
        return x, p

    def HMC_step(self, x_old):
        def H(x, p):
            return -self.log_prob(x) + 0.5 * torch.sum(p.pow(2), dim=-1)

        p_old = torch.randn_like(x_old)
        x_new, p_new = self.leapfrog(x_old.clone(), p_old.clone())
        log_accept_prob = -(H(x_new, p_new) - H(x_old, p_old))
        log_accept_prob[log_accept_prob > 0] = 0

        accept = torch.log(torch.rand_like(log_accept_prob)) < log_accept_prob
        accept = accept.unsqueeze(dim=-1)
        ret = (
            x_new * accept + x_old * torch.logical_not(accept),
            accept.sum() / accept.numel(),
        )
        return ret

    def forward(self, x):
        accept_probs = []
        samples = []
        for _ in tqdm(range(self.T)):
            x, accept_prob = self.HMC_step(x)
            accept_probs.append(accept_prob)
            samples.append(x.clone().detach())
        accept_prob = torch.mean(torch.tensor(accept_probs))
        # return last 1000 samples of HMC chain
        return torch.stack(samples)[-1000:, :], accept_prob, accept_probs

def make_log_prob_flat(ys_train, Psi, tau_sq=1.0, a_sigma=2.0, b_sigma=2.0):
    L = Psi.shape[1]  # number of covariates

    def log_prob_flat(x):
        # unpack parameters
        beta = x[:L]
        lambda_j = x[L:2*L]
        sigma2 = x[-1]

        return log_posterior_horseshoe(
            ys_train, Psi, beta, L, sigma2, tau_sq, lambda_j, a_sigma, b_sigma
        )

    return log_prob_flat

def unpack_samples(samples, L):
    w_samples = samples[:, :L]
    lambda_samples = torch.exp(samples[:, L:2*L])
    sigma_eps_sq_samples = torch.exp(samples[:, -1])
    return w_samples, lambda_samples, sigma_eps_sq_samples

def log_posterior_horseshoe(ys_train, Psi, w, N, log_sigma_eps_sq, log_tau_sq, log_lambdas, a_sigma = 2, b_sigma = 2):

    sigma_eps_sq = torch.exp(log_sigma_eps_sq) #+ 1e-5
    tau_sq = math.exp(log_tau_sq) #+ 1e-5
    lambdas = torch.exp(log_lambdas) #+ 1e-5

    ys_pred = (Psi @ w)

    log_likelihood =  -0.5 * N * torch.log(sigma_eps_sq) -0.5 * torch.sum((ys_train - ys_pred.unsqueeze(-1)) ** 2) / sigma_eps_sq

    log_p_w = -0.5 * torch.sum(w**2 / (tau_sq * lambdas**2)) - torch.sum(torch.log(lambdas))
    log_p_lambdas = torch.sum(torch.log(2/(torch.pi * (1 + lambdas**2))))

    log_p_sigma = -(a_sigma+1)*torch.log(sigma_eps_sq) - b_sigma / sigma_eps_sq
    
    return log_likelihood +  log_p_w + log_p_sigma + log_p_lambdas