import torch
from torch import nn


def J_(N: int) -> torch.tensor:
    """Get symplectic operator

    Args:
        N (int): even number, size of symplectic matrix

    Returns:
        torch.tensor: symplectic matrix with shape (N,N)
    """
    neg_I = -torch.diag(torch.ones(int(N/2)))
    pos_I = torch.diag(torch.ones(int(N/2)))
    J = torch.zeros(N, N)
    J[:int(N/2), int(N/2):] = pos_I
    J[int(N/2):, :int(N/2)] = neg_I
    return J


class sp(nn.Module):
    def __init__(self, size: int):
        """
        real symplectic lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (2n,2n ).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        if size % 2 == 0:
            self.size = size
        else:
            raise ValueError(
                'size of symplectic lie algebra matrix needs to be an even number')

        self.J = J_(size)

    @staticmethod
    def frame(X: torch.tensor, J: torch.tensor) -> torch.tensor:
        """ parametrise real symplectic lie algebra from the gneal linear matrix X

        Args:
            X (torch.tensor): (...,2n,2n)
            J (torch.tensor): (2n,2n), symplectic operator [[0,I],[-I,0]]

        Returns:
            torch.tensor: (...,2n,2n)
        """
        X = (X + X.transpose(-2, -1))/2

        return X.matmul(J.T)

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not sqaured matrix')
        return self.frame(X, self.J.to(X.device))

    @staticmethod
    def in_lie_algebra(X, eps=1e-5):
        J = J_(N=X.shape[-1])
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(J.mm(X.permute(-2, -1, 0)
                                        ).permute(-1, 0, 1)+X.mm(J.T), 0, atol=eps))
