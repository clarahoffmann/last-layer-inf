"""SG-MCMC for post-hoc and full training."""

import torch
import math
from tqdm import tqdm

def train_sg_mcmc_gauss(Psi, lli_net, dataloader_train, batch_size, sigma_eps_sq, a, b, gamma, T, warm_up):
    
    w  = torch.ones(Psi.shape[1])*0.01
    
    epsilon_t = 0.01
    epsilon_ts = []
    #N = len(dataloader_train.dataset)
    samples = []
    for t in tqdm(range(T)):
        for batch_idx, train_data in enumerate(dataloader_train):

            xs_batch, ys_batch = train_data

            with torch.no_grad():
                Psi_batch = lli_net.get_ll_embedd(xs_batch)

            ys_pred = (Psi_batch @ w).unsqueeze(-1)

            # log prior + log likelihood + noise
            # l2 norma for Gaussian log prior
            d_log_prior = - w
            
            # gradient of log-likelihood
            residual = ys_batch - ys_pred
            d_log_likelihood = (Psi_batch.T @ residual)/sigma_eps_sq
            # scale with batch size bc gradients explode 
            # if scaling with N/batch_size.
            d_log_likelihood *= 1 / batch_size 

            # full gradient
            d_full = d_log_likelihood + d_log_prior.reshape(-1,1)
            d_full = torch.clamp(d_full, min=-10, max=10)
            
            # noise for sampling
            eta = (torch.randn_like(w)* math.sqrt(epsilon_t)).reshape(-1,1)

            # update last-layer weights w_t
            w += (0.5*epsilon_t*d_full+ eta).squeeze()

            # update learning rate
            epsilon_t = a*(b + t)**(-gamma)

            if t > warm_up:
                samples.append(w.clone())
                epsilon_ts.append(epsilon_t)
    
    return samples, epsilon_ts
    
def predict_sg_mcmc_gauss(lli_net, xs_val, w_sample, epsilon_ts, sigma_eps):

    with torch.no_grad():
        Psi_val = lli_net.get_ll_embedd(xs_val)

    thin = 5
    pred_mus = []
    # use a weighted average for the predictive mean as in eq. 11
    # in Welling, Teh 2011.
    for w_sample, epsilon_t in zip(w_sample[::thin], epsilon_ts[::thin]):
        # weight with learning rate/noise variance epsilon_t
        pred_mu = (Psi_val @ w_sample)*epsilon_t
        pred_mus.append(pred_mu.unsqueeze(-1))

    pred_mus = torch.cat(pred_mus, dim = 1)
    # divide by the sum over all learning rates.
    pred_mu = pred_mus.sum(dim = 1)/torch.sum(epsilon_ts[::thin])

    pred_sigma = torch.sqrt(pred_mus.var(dim = 1) + sigma_eps**2)

    return pred_mu, pred_sigma

