"""SG-MCMC for post-hoc and full training."""

import torch

def train_sg_mcmc(model, xs, ys, lr=1e-2, noise_std=1e-2, n_steps=3000, burn_in=1000):
    samples = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()
        preds = model.forward(xs)
        loss = ((preds - ys)**2).mean()
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p) * (2 * lr * noise_std)**0.5
                    p.add_(-lr * p.grad + noise)

        if step >= burn_in and step % 10 == 0:
            samples.append({k: v.detach().clone() for k, v in model.state_dict().items()})

    return samples

