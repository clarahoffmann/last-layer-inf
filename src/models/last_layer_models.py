"""Last-layer inference models."""
import torch
from torch import nn
from tqdm import tqdm 
from torch.optim import Adam
from typing import Tuple
import numpy as np

MANUAL_SEED = 758
DETERMINISTIC_TRAINING = True

class LLI(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
       
        self.fc = nn.Sequential(
            *[
                layer
                for in_dim, out_dim in zip(dims[:-2], dims[1:-1])
                for layer in (nn.Linear(in_dim, out_dim), nn.ReLU())
            ],
        )
        self.last_layer = nn.Linear(dims[-2], dims[-1])
    
    def forward(self, x):
        x = self.fc(x)
        x = self.last_layer(x)
        return x
    
    def get_ll_embedd(self, x):
        x = self.fc(x)
        return x
    

def train_last_layer_det(model: nn.Module, 
                         dataloader_train: torch.utils.data.DataLoader, 
                         dataloader_val: torch.utils.data.DataLoader, 
                         weight_decay: float = 0,
                         num_epochs: int = 100, lr = 1e-3) -> Tuple[nn.Module, list, list]:

    torch.manual_seed(MANUAL_SEED)
    torch.use_deterministic_algorithms(DETERMINISTIC_TRAINING)
    
    # define loss and optimizer
    optimizer = Adam(lr = lr, params = model.parameters(), weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    losses_train = []
    losses_val = []
    # start training
    for _ in tqdm(range(num_epochs)):
        
        # training step
        loss_train = 0
        for x, y in dataloader_train:
            optimizer.zero_grad()
            
            y_pred = model.forward(x)
            
            loss = loss_fn(y, y_pred)  
            loss.backward()

            optimizer.step()

            loss_train += loss.item() * x.size(0)
        
        losses_train.append(loss_train/ len(dataloader_train.dataset))
        
        
        # val step
        val_loss = 0
        for x_val, y_val in dataloader_val:
            
            with torch.no_grad():
                y_pred_val = model.forward(x_val)
                loss = loss_fn(y_val, y_pred_val)  
                val_loss += loss.item() * x_val.size(0)
        
        losses_val.append(val_loss / len(dataloader_val.dataset))

    return model, losses_train, losses_val

def fit_last_layer_closed_form(model_dims, dataloader_train, dataloader_val, xs_train, ys_train, xs_val, num_epochs, sigma_0):

    # fit deep feature projector
    lli_net = LLI(model_dims)
    lli_net, losses_train, losses_val = train_last_layer_det(model = lli_net, 
                                                        dataloader_train = dataloader_train,
                                                        dataloader_val = dataloader_val, 
                                                        num_epochs = num_epochs)
    
    lli_net.eval()

    # fit last-layer posterior
    with torch.no_grad():
        Psi = lli_net.get_ll_embedd(xs_train).detach()
        d = Psi.shape[1]
        Sigma_N_inv = (1/sigma_0**2)*torch.eye(d) + (1/sigma_0**2)*(Psi.T @ Psi)
        Sigma_N = torch.linalg.inv(Sigma_N_inv).detach()
        mu_N = Sigma_N @ ((1/sigma_0**2)*(Psi.T @ ys_train))
        Sigma_N = Sigma_N

    # get pred. means and variances
    lli_pred_mu, lli_pred_sigma_sq = get_post_pred_dens(model = lli_net, x_star = xs_val , 
                              mu_N = mu_N, Sigma_N = Sigma_N, sigma_eps = sigma_0 )
    
    out_dict = {'mu_N': mu_N.detach().numpy(),
           'Sigma_N': Sigma_N.detach().numpy(),
           'pred_mu': lli_pred_mu,
           'pred_sigma': np.sqrt(lli_pred_sigma_sq),
           'losses_train': losses_train,
           'losses_val': losses_val,
           }
    
    return out_dict, lli_net

def get_metrics_lli_closed_form(mu_N, Sigma_N, model, xs_val, ys_val, sigma_0 ):
    lli_pred_mu, lli_pred_sigma_sq = get_post_pred_dens(model = model, x_star = xs_val , 
                              mu_N = mu_N, Sigma_N = Sigma_N, sigma_eps = sigma_0 )
    lli_pred_sigma = np.sqrt(lli_pred_sigma_sq)

    rmse_lli = (ys_val.reshape(-1,1) - torch.tensor(lli_pred_mu)).pow(2).sqrt()
    rmse_lli_mean = rmse_lli.mean()
    rmse_lli_std = rmse_lli.std()

    return rmse_lli_mean, rmse_lli_std, lli_pred_mu, lli_pred_sigma
    


def get_post_pred_dens(model: nn.Module, x_star: np.ndarray, 
                       mu_N: torch.tensor, Sigma_N: torch.tensor, 
                       sigma_eps: float):
    psi_stars = model.get_ll_embedd(x_star).detach() #.reshape(1,1)
    mu_star = mu_N.T @ psi_stars.T
    sigma_sq_star = np.array([sigma_eps**2 + psi_star @ Sigma_N @ psi_star.T for psi_star in psi_stars])
    return mu_star.detach().squeeze().numpy(), sigma_sq_star


class LastLayerVIClosedForm(nn.Module):
    def __init__(self, dim_last_layer, dim_output, prior_var=.3):
        super().__init__()
        self.dim_last_layer = dim_last_layer
        self.dim_output = dim_output
        
        self.prior_var = prior_var
        self.prior_cov = torch.eye(dim_last_layer) * prior_var
        self.prior_cov_inv = torch.inverse(self.prior_cov)
        
        self.mu = nn.Parameter(torch.zeros(dim_output, dim_last_layer))
        self.L_unconstrained = nn.Parameter(torch.randn(dim_output, dim_last_layer, dim_last_layer) * 0.01)
    
    def get_L(self):
        L = torch.tril(self.L_unconstrained)
        diag_idx = torch.arange(self.dim_last_layer)
        L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(L[:, diag_idx, diag_idx]) + 1e-5
        return L
    
    def forward_det(self, X):
        return X @ self.mu.T
    
    def forward_sample(self, X):
        L = self.get_L()
        eps = torch.randn_like(self.mu)
        w_samples = self.mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return X @ w_samples.t(), w_samples # variance is missing here
    
    def kl_divergence(self):
        L = self.get_L()
        Sigma_q = torch.bmm(L, L)
        
        log_det_prior = self.dim_last_layer * torch.log(torch.tensor(self.prior_var))
        log_det_q = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)
        
        mu = self.mu
        
        trace_term = torch.sum(mu @ self.prior_cov_inv * mu, dim=1) + \
                     torch.sum(Sigma_q * self.prior_cov_inv.T, dim=(1, 2))

        
        kl = 0.5 * (log_det_prior - log_det_q - self.dim_last_layer + trace_term)
        return kl.sum()