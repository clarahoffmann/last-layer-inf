from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torch
import torch.nn as nn
from tqdm import tqdm


@variational_estimator
class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blinear1 = BayesianLinear(1, 100)
        self.ReLU = nn.ReLU()
        self.blinear2 = BayesianLinear(100, 1)

    def forward(self, x):
        x = self.ReLU(self.blinear1(x))
        x = self.blinear2(x)
        return x
    
def train_bnn(model, num_epochs, dataloader_train, dataloader_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses_train = []
    losses_val = []
    for _ in tqdm(range(num_epochs)):
        
        loss_train = 0
        for x, y in dataloader_train:
            optimizer.zero_grad()
            loss = model.sample_elbo(
                inputs=x,
                labels=y,
                criterion=nn.MSELoss(),
                sample_nbr=3, # number MC samples
                complexity_cost_weight=1e-3 # weight of KL-divergence
            )
            loss.backward()
            optimizer.step()
            loss_train += loss.item() * x.size(0)

        losses_train.append(loss_train/ len(dataloader_train.dataset))
        
        
        val_loss = 0
        for x_val, y_val in dataloader_val:
            
            with torch.no_grad():
                loss = model.sample_elbo(
                    inputs=x_val,
                    labels=y_val,
                    criterion=nn.MSELoss(),
                    sample_nbr=3,
                    complexity_cost_weight=1e-3
                )
                val_loss += loss.item() * x_val.size(0)
            
        losses_val.append(val_loss / len(dataloader_val.dataset))
    
    return model, losses_train, losses_val



