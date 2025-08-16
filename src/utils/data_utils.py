import math
import numpy as np
import torch
from ucimlrepo import fetch_ucirepo

def f(x: float, noise: bool = True, sigma_eps: float = 0.1):
        """Generates a sample y_i from y = sin(x) + varepsilon."""
        y = math.sin(x)
        if noise:
            y += sigma_eps*np.random.randn(1)
        return y


def create_synthetic_train_data(xs_range = [-4,4], num_points = 200,  sigma_eps = 0.3):

    # draw data
    xs = np.linspace(xs_range[0], xs_range[1], num_points)
    ys = np.array([f(x = x, noise = True, sigma_eps = sigma_eps) for x in xs])

    # sample train indices
    train_idx = np.random.choice(len(xs), size=100, replace=False)

    xs_train = torch.tensor(xs[train_idx]).unsqueeze(-1).float()
    ys_train = torch.tensor(ys[train_idx]).float()

    all_idx = np.arange(len(xs))
    val_idx = np.setdiff1d(all_idx, train_idx)

    xs_val = torch.tensor(xs[val_idx]).unsqueeze(-1).float()
    ys_val = torch.tensor(ys[val_idx]).float()

    return xs, ys, xs_train, ys_train, xs_val, ys_val


def create_concrete_train_data(num_train_points = 700):
    """Creates a training, validation split for the UCI
    concrete compression strength data. The dataset comprises 1038
    training points, from which num_train_points are used for training
    and the rest for validation."""
    # Download data
    dataset = fetch_ucirepo(name="Concrete Compressive Strength")
    xs = np.array(dataset.data.features)
    ys = np.array(dataset.data.targets)
    xs = (xs - xs.mean(axis=0))/xs.std(axis=0)
    ys = (ys - ys.mean(axis=0))/ys.std(axis=0)

    # choose 100 random train indices
    train_idx = np.random.choice(len(ys), size=num_train_points, replace=False)
    xs_train = torch.tensor(xs[train_idx, :]).float()
    ys_train = torch.tensor(ys[train_idx, :]).float()

    # choose the remaining ones as validation data
    all_idx = np.arange(len(xs))
    val_idx = np.setdiff1d(all_idx, train_idx)

    xs_val = torch.tensor(xs[val_idx]).float()
    ys_val = torch.tensor(ys[val_idx]).float()
    
    return xs, ys, xs_train, ys_train, xs_val, ys_val
     