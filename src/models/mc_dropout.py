"""MC-dropout model."""
import torch
from torch import nn
from models.last_layer_models import train_last_layer_det
from utils.coverage import get_coverage_y_hats


class MCDropoutNet(nn.Module,):
    def __init__(self, dims: list[int], p: float = 0.2 ):
        super().__init__()

        self.fc = nn.Sequential(
            *[
                layer
                for in_dim, out_dim in zip(dims[:-2], dims[1:-1])
                for layer in (nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(p=p))
            ],
            nn.Linear(dims[-2], dims[-1])
        )
    def forward(self, x: torch.tensor):
        return self.fc(x)
    
def train_mc_dropout(model_dims, dataloader_train, dataloader_val, num_epochs ):
    mc_net = MCDropoutNet(model_dims, p = 0.2)

    mc_net, losses_train, losses_val = train_last_layer_det(model = mc_net, 
        dataloader_train = dataloader_train,
        dataloader_val = dataloader_val, 
        num_epochs = num_epochs)

    return mc_net.eval(), losses_train, losses_val

def fit_mc_dropout(model_dims, xs_pred, dataloader_train, dataloader_val, num_epochs ):
    
    mc_net, losses_train, losses_val =  train_mc_dropout(model_dims, dataloader_train, dataloader_val, num_epochs )
    
    # MC dropout
    num_samples_mc_dropout = 100
    preds = []
    with torch.no_grad():
        for _ in range(num_samples_mc_dropout):
            preds.append(mc_net.forward(xs_pred))

    ys_samples_mc = torch.stack(preds)
    mc_pred_mu = torch.mean(ys_samples_mc, axis = 0).squeeze()
    mc_pred_sigma = torch.std(ys_samples_mc, axis = 0).squeeze()

    out_dict = {
           'pred_mu': mc_pred_mu,
           'pred_sigma': mc_pred_sigma,
           'losses_train': losses_train,
           'losses_val': losses_val}

    return mc_net, out_dict

def get_metrics_mc_dropout(model, xs_val, ys_val):
    num_samples_mc_dropout = 100
    preds = []
    with torch.no_grad():
        for _ in range(num_samples_mc_dropout):
            preds.append(model.forward(xs_val))

    ys_samples_mc = torch.stack(preds)
    mc_pred_mu = torch.mean(ys_samples_mc, axis = 0).squeeze()
    mc_pred_sigma = torch.std(ys_samples_mc, axis = 0).squeeze()

    rmse_mc = (ys_val.reshape(-1,1) - mc_pred_mu).pow(2).sqrt()
    rmse_mc_mean = rmse_mc.mean()
    rmse_mc_std = rmse_mc.std()

    return rmse_mc_mean, rmse_mc_std, mc_pred_mu, mc_pred_sigma, ys_samples_mc

def get_coverage_mc_dropout(ys_samples_mc, ys_val, levels):
    coverage_mc_dropout = get_coverage_y_hats(ys_samples_mc, ys_val, levels)
    return coverage_mc_dropout
