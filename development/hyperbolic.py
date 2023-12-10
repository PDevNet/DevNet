import torch
from torch import nn


class hyperbolic(nn.Module):
    def __init__(self, size):
        """
        lie algebra matrices preserving orientation of the hyperbolic symmetris,
        parametrized in terms of 
        by a general linear matrix with shape (...,...,n,n).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """parametrise lie algebra matrices preserving orientation of the hyperbolic symmetris
             from the general linear matrix X

        Args:
            X (torch.tensor): (...,2n,2n)

        Returns:
            torch.tensor: (...,2n,2n)
        """
        N, C, m, m = X.shape
        g = torch.eye(m)
        g[-1, -1] = -1
        A = X - g @ X.transpose(-2, -1) @ g

        return A


    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        return self.frame(X)
