"""
Adapted from https://github.com/Lezcano/geotorch/blob/master/geotorch/so.py
"""
import torch
from torch import nn


class so(nn.Module):
    def __init__(self, size):
        """
        so(n) lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (...,...,n,n).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.size = size

    @staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """ parametrise real symplectic lie algebra from the gneal linear matrix X

        Args:
            X (torch.tensor): (...,2n,2n)
            J (torch.tensor): (2n,2n), symplectic operator [[0,I],[-I,0]]

        Returns:
            torch.tensor: (...,2n,2n)
        """

        X = X.tril(-1)
        X = X - X.transpose(-2, -1)

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not sqaured matrix')
        return self.frame(X)

    @ staticmethod
    def in_lie_algebra(X, eps=1e-5):
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(X.transpose(-2, -1), -X, atol=eps))


def so_uniform_init_(tensor):
    r"""Fills in the input ``tensor`` in place with an orthogonal matrix.
    If square, the matrix will have positive determinant.
    The tensor will be distributed according to the Haar measure.
    The input tensor must have at least 2 dimensions.
    For tensors with more than 2 dimensions the first dimensions are treated as
    batch dimensions.
    Args:
        tensor (torch.Tensor): a 2-dimensional tensor or a batch of them
    """
    # We re-implement torch.nn.init.orthogonal_, as their treatment of batches
    # is not in a per-matrix base
    if tensor.ndim < 2:
        raise ValueError(
            "Only tensors with 2 or more dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )
    n, k = tensor.size()[-2:]
    transpose = n < k
    with torch.no_grad():
        x = torch.empty_like(tensor).normal_(0, 1)
        if transpose:
            x.transpose_(-2, -1)
        q, r = torch.linalg.qr(x)

        # Make uniform (diag r >= 0)
        d = r.diagonal(dim1=-2, dim2=-1).sign()
        q *= d.unsqueeze(-2)
        if transpose:
            q.transpose_(-2, -1)

        # Make them have positive determinant by multiplying the
        # first column by -1 (does not change the measure)
        if n == k:
            mask = (torch.det(q) >= 0.0).float()
            mask[mask == 0.0] = -1.0
            mask = mask.unsqueeze(-1)
            q[..., 0] *= mask
        tensor.copy_(q)
        return tensor
