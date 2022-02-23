import torch
from torch import nn


class se(nn.Module):
    def __init__(self, size):
        """
        se(n) lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (...,...,n,n).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """ parametrise special euclidean lie algebra from the gneal linear matrix X

        Args:
            X (torch.tensor): (...,2n,2n)

        Returns:
            torch.tensor: (...,2n,2n)
        """
        N, C, m, m = X.shape
        so = X[..., :-1, :-1] - X[..., :-1, :-1].transpose(-2, -1)

        X = torch.cat(
            (torch.cat((so, X[..., :-1, -1].unsqueeze(-1)), dim=3
                       ), torch.zeros((N, C, 1, m)).to(X.device)), dim=2)

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not sqaured matrix')
        return self.frame(X)
