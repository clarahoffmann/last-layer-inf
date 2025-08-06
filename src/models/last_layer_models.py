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
                         num_epochs: int = 100) -> Tuple[nn.Module, list, list]:

    torch.manual_seed(MANUAL_SEED)
    torch.use_deterministic_algorithms(DETERMINISTIC_TRAINING)
    
    # define loss and optimizer
    optimizer = Adam(lr = 1e-3, params = model.parameters(), weight_decay=weight_decay)
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


def get_post_pred_dens(model: nn.Module, x_star: np.ndarray, 
                       mu_N: torch.tensor, Sigma_N: torch.tensor, 
                       sigma_eps: float):
    psi_stars = model.get_ll_embedd(x_star).detach() #.reshape(1,1)
    mu_star = mu_N.T @ psi_stars.T
    sigma_sq_star = np.array([sigma_eps**2 + psi_star @ Sigma_N @ psi_star.T for psi_star in psi_stars])
    return mu_star.detach().squeeze().numpy(), sigma_sq_star