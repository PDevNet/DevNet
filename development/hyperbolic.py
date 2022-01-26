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
        """ parametrise lie algebra matrices preserving orientation of the hyperbolic symmetris
             from the gneal linear matrix X

        Args:
            X (torch.tensor): (...,2n,2n)

        Returns:
            torch.tensor: (...,2n,2n)
        """
        N, C, m, m = X.shape
        X = torch.cat([torch.zeros(N, C, m-1, m-1),
                      X[..., :-1, -1].unsqueeze(-1)], dim=-1)
        X = torch.cat([X, torch.zeros(N, C, 1, m)], dim=-2)

        X = X + X.transpose(-2, -1)

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        return self.frame(X)
