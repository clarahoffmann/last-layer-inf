from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torch
import torch.nn as nn
from tqdm import tqdm



"""class BNN(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc = nn.Sequential(
            *[
                layer
                for in_dim, out_dim in zip(dims[:-2], dims[1:-1])
                for layer in (BayesianLinear(in_dim, out_dim), nn.ReLU())
            ],
            nn.Linear(dims[-2], dims[-1])
        )

    def forward(self, x):
        x = self.fc(x)
        return x"""

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


def fit_bnn(num_epochs, xs_pred,  dataloader_train, dataloader_val):
    bnn = BNN()
    bnn, losses_train, losses_val = train_bnn(model = bnn, 
                                          num_epochs = num_epochs*2, 
                                          dataloader_train = dataloader_train, 
                                          dataloader_val = dataloader_val)
    
    bnn_samples = []
    with torch.no_grad():
        preds = [bnn(xs_pred) for _ in range(100)]  # 100 MC samples
        preds = torch.stack(preds)
        bnn_pred_mu = preds.mean(dim=0).squeeze()
        bnn_pred_std = preds.std(dim=0).squeeze()
        bnn_samples.append(preds)
    
    out_dict = {'pred_mu': bnn_pred_mu,
           'pred_sigma': bnn_pred_std,
           'losses_train': losses_train,
           'losses_val': losses_val,
           }
    
    return bnn.eval(), out_dict
    

def get_metrics_bnn(bnn, xs_val, ys_val):
    bnn_samples = []
    with torch.no_grad():
        preds = [bnn(xs_val) for _ in range(100)]  # 100 MC samples
        preds = torch.stack(preds)
        bnn_pred_mu = preds.mean(dim=0).squeeze()
        bnn_pred_std = preds.std(dim=0).squeeze()
        bnn_samples.append(preds)

    rmse_bnn = (ys_val.reshape(-1,1) - bnn_pred_mu).pow(2).sqrt()
    rmse_bnn_mean = rmse_bnn.mean()
    rmse_bnn_std = rmse_bnn.std()

    return bnn_samples, rmse_bnn_mean, rmse_bnn_std


