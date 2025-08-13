"""Variational inference for last-layer inference models."""
import math
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, Literal
from scipy.special import gammaln
from torch.optim import Adam

def kl_w_vectorized_full_cov(mu_samples, Sigma_q, tau_sq_samples):
    L = Sigma_q.shape[0]
    
    # log determinant of Sigma_q
    logdet_Sigma = torch.logdet(Sigma_q + 1e-5 * torch.eye(L))
    
    # trace of Sigma_q / tau_sq_samples
    trace_term = torch.trace(Sigma_q) / tau_sq_samples
    mu_norm_sq = torch.sum(mu_samples**2, dim=1) / tau_sq_samples
    log_tau_sq = torch.log(tau_sq_samples)
    
    kl = 0.5 * (trace_term + mu_norm_sq - L + L * log_tau_sq - logdet_Sigma)
    
    return kl.mean()

def kl_w_vectorized(mu, Sigma_q, tau_sq_samples):
    _, L = mu.shape

    logdet_Sigma = torch.sum(torch.log(Sigma_q), dim=1)
    trace_term = torch.sum(Sigma_q, dim=1)              
    mu_norm_sq = torch.sum(mu**2, dim=1)                
    log_tau_sq = torch.log(tau_sq_samples)               

    kl = 0.5 * (trace_term / tau_sq_samples + mu_norm_sq / tau_sq_samples - L + L * log_tau_sq - logdet_Sigma)

    return kl.mean()

def kl_tau_sq(q_log_pdf_normal_value, tau_sq_samples, a_tau, b_tau):
    log_prior_ig = (
        a_tau * torch.log(torch.tensor(b_tau)) 
        - torch.lgamma(torch.tensor(a_tau))
        - (a_tau + 1) * torch.log(tau_sq_samples) 
        - b_tau / tau_sq_samples
    )
    
    kl = q_log_pdf_normal_value.mean() - log_prior_ig.mean()
    return kl

def q_log_pdf_lognormal(sigma_eps_sq_samples, mu_log, log_var_log):

    var_log = torch.exp(log_var_log)
    log_x = torch.log(sigma_eps_sq_samples)
    log_pdf = (
        - log_x
        - 0.5 * torch.log(2 * torch.pi * var_log)
        - (log_x - mu_log)**2 / (2 * var_log)
    )
    return log_pdf

def log_p_sigma_eps_sq_ig(sigma_eps_sq, a_sigma, b_sigma):
    log_prob1 = a_sigma * math.log(b_sigma) - gammaln(a_sigma)
    log_prob2 = -(a_sigma + 1) * torch.log(sigma_eps_sq) - b_sigma / sigma_eps_sq
    return log_prob1 + log_prob2  # shape [S]

def kl_sigma_eps_sq(q_log_pdf, sigma_eps_sq_samples, a_sigma, b_sigma):
    log_prior = log_p_sigma_eps_sq_ig(sigma_eps_sq_samples, a_sigma, b_sigma)
    return (q_log_pdf - log_prior).mean()
    


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


def run_last_layer_vi_ridge(model, Psi, ys_train, optimizer_vi, num_epochs = 30000):

    N, _ = Psi.shape

    elbos = []
    for epoch in range(num_epochs):
        optimizer_vi.zero_grad()
        
        params_samples, params, y_sample, Sigma_q = model.forward(Psi)

        # samples of parameters
        w_samples =  params_samples[:, :model.in_features]
        sigma_eps_sq_samples = torch.exp(params_samples[:, -1]) + 1e-5
        tau_sq_samples = torch.exp(params_samples[:, -2]) + 1e-5

        # variational means
        q_w_mu = params[:, :model.in_features]
        q_log_tau_sq_mu = params_samples[:, -2]
        q_log_sigma_eps_sq_mu = params_samples[:, -1]

        q_log_sigma_eps_sq_var = Sigma_q[:, -1]
        q_log_tau_sq_var = Sigma_q[:, -2]

        # likelihood
        log_likelihood = (
            - 0.5 * N * torch.log(sigma_eps_sq_samples) 
            - 0.5 * (torch.sum((ys_train - y_sample) ** 2)/ sigma_eps_sq_samples))
        
        # expected KL
        kl_w = kl_w_vectorized(w_samples, Sigma_q[:,:model.in_features], tau_sq_samples)
        q_log_pdf_normal_value_eps = q_log_pdf_lognormal(sigma_eps_sq_samples, q_log_sigma_eps_sq_mu, q_log_sigma_eps_sq_var)
        
        kl_sigma_eps_sq_value = kl_sigma_eps_sq(q_log_pdf_normal_value_eps, sigma_eps_sq_samples, 2, 2)
        q_log_pdf_normal_value_tau_value = q_log_pdf_lognormal(tau_sq_samples, q_log_tau_sq_mu, q_log_tau_sq_var)
        kl_tau_sq_value = kl_tau_sq(q_log_pdf_normal_value_tau_value, tau_sq_samples, 2, 2)
        
        elbo = (
            log_likelihood.mean() 
            - kl_w - kl_sigma_eps_sq_value - kl_tau_sq_value
        )
        loss = -elbo
        loss.backward()
        optimizer_vi.step()
        if epoch % 1000 == 0:
            print(f"VI epoch {epoch} ELBO: {elbo.item():.3f}")

        elbos.append(elbo.item())

    return model, elbos


def run_last_layer_vi_horseshoe(model, Psi, ys_train, optimizer_vi, num_epochs = 30000, lr=1e-3):
    
    elbos = []
    N, _ = Psi.shape

    for epoch in range(num_epochs):
        optimizer_vi.zero_grad()
        
        # Get var. parameters + samples
        params_samples, params, y_sample, Sigma_q = model.forward(Psi)

        # Sample of.
        # last-layer weights
        w_samples = params_samples[:, :model.in_features]
        # variance of obs. noise 
        sigma_eps_sq_samples = torch.exp(params_samples[:, -1]) + 1e-5 
        # Local shrinkage
        lambda_samples = torch.exp(params_samples[:, model.in_features:-1]) + 1e-8

        # Variational means and variances
        q_w_mu = params[:, :model.in_features]
        q_lambda_mu = params[:, model.in_features:-1]
        q_lambda_var = Sigma_q[:, model.in_features:-1]

        q_log_sigma_eps_sq_mu = params_samples[:, -1]
        q_log_sigma_eps_sq_var = Sigma_q[:, -1]

        # Likelihood 
        log_likelihood = (
            -0.5 * N * torch.log(sigma_eps_sq_samples.unsqueeze(0))
            -0.5 * (ys_train - y_sample) ** 2 / sigma_eps_sq_samples.unsqueeze(0))
        ll_term = log_likelihood.sum(dim=0).mean()

        # KL term for w
        Sigma_w_diag = Sigma_q[:, :model.in_features]
        E_inv_lambda_sq = torch.exp(-2 * q_lambda_mu + 2 * q_lambda_var)
        E_log_lambda_sq = 2 * q_lambda_mu
        kl_w = 0.5 * torch.sum(
            (Sigma_w_diag.mean(dim=0) + q_w_mu.mean(dim=0)**2) * E_inv_lambda_sq.mean(dim=0)
            - 1
            + E_log_lambda_sq.mean(dim=0)
            - torch.log(Sigma_w_diag.mean(dim=0) + 1e-8)
        )

        # KL term for local shrinkage lambda
        q_log_pdf_lambda = q_log_pdf_lognormal(lambda_samples, q_lambda_mu, q_lambda_var)
        log_p_lambda = math.log(2.0 / math.pi) - torch.log(1.0 + lambda_samples**2)
        kl_lambda = torch.sum((q_log_pdf_lambda - log_p_lambda).mean(dim=0))

        # KL term for sigma_eps_sq
        q_log_pdf_sigma_eps = q_log_pdf_lognormal(sigma_eps_sq_samples,
                                                q_log_sigma_eps_sq_mu,
                                                q_log_sigma_eps_sq_var)
        
        q_log_pdf_normal_value_eps = q_log_pdf_lognormal(sigma_eps_sq_samples, q_log_sigma_eps_sq_mu, q_log_sigma_eps_sq_var)
        kl_sigma_eps_sq_value = kl_sigma_eps_sq(q_log_pdf_normal_value_eps, sigma_eps_sq_samples, 2, 2)

        # ELBO
        elbo = ll_term - kl_w - kl_lambda - kl_sigma_eps_sq_value
        loss = -elbo

        loss.backward()
        optimizer_vi.step()

        if epoch % 100== 0:
            print(f"VI epoch {epoch} ELBO: {elbo.item():.3f}")

        elbos.append(elbo.item())

    return model, elbos


def run_last_layer_vi_ridge_full_fac(model, Psi, ys_train, optimizer_vi, num_epochs = 20000):

    elbos = []
    N, _ = Psi.shape
    for epoch in range(num_epochs):
        optimizer_vi.zero_grad()
        
        # forward pass
        params_samples, params, y_sample, Sigma_q = model.forward(Psi)

        # sample parameters
        w_samples = params_samples[:, :model.in_features]
        tau_sq_samples = torch.nn.functional.softplus(params_samples[:, -2]) + 1e-5
        sigma_eps_sq_samples = (torch.nn.functional.softplus(params_samples[:, -1]) + 1e-5) #.unsqueeze(0)

        # variational means
        q_w_mu = params[:, :model.in_features]
        q_log_tau_sq_mu = params[:, -2]
        q_log_sigma_eps_sq_mu = params[:, -1]

        # variances from Sigma_q
        q_log_tau_sq_var = Sigma_q[-2, -2]
        q_log_sigma_eps_sq_var = Sigma_q[-1, -1]

        # likelihood
        log_likelihood = -0.5 * N * torch.log(sigma_eps_sq_samples) \
                        -0.5 * torch.sum((ys_train.squeeze().unsqueeze(0) - y_sample)**2, dim = 1) / sigma_eps_sq_samples
        log_likelihood = log_likelihood.mean()

        # KL terms
        kl_w = kl_w_vectorized_full_cov(w_samples, Sigma_q[:model.in_features, :model.in_features], tau_sq_samples)
        
        q_log_pdf_eps = q_log_pdf_lognormal(sigma_eps_sq_samples, q_log_sigma_eps_sq_mu, q_log_sigma_eps_sq_var)
        kl_sigma_eps = kl_sigma_eps_sq(q_log_pdf_eps, sigma_eps_sq_samples, a_sigma=2, b_sigma=2)
        
        q_log_pdf_tau = q_log_pdf_lognormal(tau_sq_samples, q_log_tau_sq_mu, q_log_tau_sq_var)
        kl_tau = kl_tau_sq(q_log_pdf_tau, tau_sq_samples, a_tau=2, b_tau=2)

        # ELBO
        elbo = log_likelihood - kl_sigma_eps - kl_tau - kl_w
        loss = -elbo

        # backward
        loss.backward()
        optimizer_vi.step()

        if epoch % 1000 == 0:
            print(f"VI epoch {epoch} ELBO: {elbo.item():.3f}")

        elbos.append(elbo.item())

    return model, elbos