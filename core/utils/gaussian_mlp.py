import torch
import torch.nn as nn


class Gaussian(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x**2) / (2 * self.sigma**2))


class PGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.exp(-(x**2) / (2 * self.sigma**2))


class GaussianMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, sigma=0.05):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.sigma = sigma

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_in, dim_hidden))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.layers.append(nn.Linear(dim_hidden, dim_out))

        self.activation = Gaussian(self.sigma)

        self.init_()

    def init_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1).astype(
        np.float32
    )
    xxyy = torch.from_numpy(xxyy)

    model = GaussianMLP(dim_in=2, dim_out=1, dim_hidden=256, num_layers=5, sigma=0.05)
    z = model(xxyy).detach().numpy().reshape(100, 100)

    plt.imshow(z)
    plt.savefig("debug/gaussian_mlp.png")
