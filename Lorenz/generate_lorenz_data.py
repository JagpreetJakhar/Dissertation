import torch
import os
from torchdiffeq import odeint

def lorenz(t, X):
    sigma = 10.
    beta = 8/3
    rho = 28.
    if X.ndim == 1:
        x, y, z = X
    else:
        x, y, z = X.unbind(dim=-1)
    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return torch.stack([xdot, ydot, zdot], dim=-1)

x_0 = torch.tensor([1., 0., 0.])

# Define the time values
t = torch.linspace(0., 1000., 1000)

# Using ODE Solver to Integrate the function to get the data
true_lorenz = odeint(lorenz, x_0, t)
torch.save(true_lorenz,'Dissertation\Data\Lorenz\lorenz_data.pt')