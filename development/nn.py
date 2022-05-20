import math
from development.so import so_uniform_init_
from development.expm import expm, rescaled_matrix_exp
import torch
import torch.nn as nn
import numpy as np
from development.sp import sp
from development.so import so
import sys
sys.argv = ['']
del sys


class projection(nn.Module):
    def __init__(self, input_size, hidden_size, channels=1, param=sp, triv=expm, **kwargs):
        """this class is used to project the path increments to the Lie group path increments, with Lie algbra trainable weights.
        Args:
            input_size (int): input size
            hidden_size (int): size of the hidden Lie algbra matrix
            channels (int, optional): number of channels, produce independent Lie algebra weights. Defaults to 1.
            param (method, optional): parametrization method to map the GL matrix to required matrix Lie algebra. Defaults to sp.
            triv (function, optional): the trivialization map from the Lie algebra to its correpsonding Lie group. Defaults to expm.
        """
        self.__dict__.update(kwargs)

        if self.complex:
            A = torch.empty(input_size, channels, hidden_size,
                            hidden_size, dtype=torch.cfloat)
        else:
            A = torch.empty(input_size, channels, hidden_size,
                            hidden_size)  # (C,m,m)
        super(projection, self).__init__()
        # self.size = hidden_size
        self.param_map = param(hidden_size)
        self.param = param
        self.A = nn.Parameter(self.param_map(A))
        # print(self.param_map.in_lie_algebra(self.A))
        self.triv = triv
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform(self.A)
        if self.param.__name__ in ['orthogonal', 'se']:
            so_uniform_init_(self.A)
            self.param_map(self.A)

        else:
            nn.init.kaiming_uniform_(self.A)
            self.param_map(self.A)

    def forward(self, dX: torch.tensor) -> torch.tensor:
        """

        Args:
            dX (torch.tensor): (N,input_size)
        Returns:
            torch.tensor: (N,channels,hidden_size,hidden_size)
        """
        # print(self.param_map.in_lie_algebra(self.A))
        # print('forward test :', self.param_map.in_lie_algebra(
        # self.param_map(self.A)))

        A = self.param_map(self.A).permute(1, 2, -1, 0)  # C,m,m,in

        # A = self.A.permute(1, 2, -1, 0)
        AX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        # X -> A := M(X)  expm(M(X))
        return rescaled_matrix_exp(self.triv, AX)
        # return self.triv(AX)


class development_layer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, channels: int = 1, param=sp,
                 triv=expm, return_sequence: bool = False, complexification: bool = False,
                 include_inital: bool = False, time_batch=1):
        """This the main development layer class, which map the input Euclidean path to the matrix Lie group valued
            path.

        Args:
            input_size (int): dimension of the input time series
            hidden_size (int): dimension of the hidden matrix Lie group,
                               will return matrix of shape hidden_size * hidden_size
            channels (int, optional): Number of channels, this is optional argument allow us to compute
                                      multiple independent developments at once. Defaults to 1.
            param (method, optional): parametrization method to map the GL matrix to required matrix Lie algebra. Defaults to sp.
            triv (function, optional): the trivialization map from the Lie algebra to its correpsonding Lie group. Defaults to expm.
            return_sequence (bool, optional): If return_sequence == True: return whole development path
                                              If return_sequence == False, only return the matrix Lie group element at the last time step
                                              Defaults to False.
            complexification (bool, optional): If True, use the complex valued Lie group
                                               If False use the real valued Lie group.  Defaults to False.
            include_inital (bool, optional): If True, include the intial state of the time series
                                             If False, dont include the intial state of the time series, development is translation invariance.
                                             Defaults to False.
        """
        super(development_layer, self).__init__()
        self.complex = complexification
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.return_sequence = return_sequence
        self.projection = projection(
            input_size, hidden_size, channels=channels, param=param, triv=triv, complex=self.complex)
        self.include_inital = include_inital
        self.truncation = time_batch

    def forward(self, input: torch.tensor) -> torch.tensor:
        """forward

        Args:
            input (torch.tensor): tensor with shape (N,T,input_size)

        Returns:
            [type]: [description] (N,T,channels,hidden_size,hidden_size)
        """
        if self.complex:
            input = input.to(torch.cfloat)

        N, T, C = input.shape
        if self.include_inital:
            input = torch.cat(
                [torch.zeros((N, 1, C)).to(input.device), input], dim=1)
        dX = input[:, 1:] - input[:, :-1]  # N,T-1,input_size
        time_len = math.ceil(T/self.truncation)
        out = torch.eye(self.hidden_size, device=input.device, dtype=input.dtype).reshape(
            1, 1, self.hidden_size, self.hidden_size).repeat(N, self.channels, 1, 1)
        if self.return_sequence:
            out = []
            for i in range(0, T, time_len):

                dX1 = dX[:, i:i+time_len].reshape(-1, dX.shape[-1])
                M_dX = self.projection(dX1).reshape(
                    N, -1, self.channels, self.hidden_size, self.hidden_size)
                out.append(self.prod(M_dX))
            return torch.cat(out, 1)
        else:
            for i in range(0, T, time_len):

                dX1 = dX[:, i:i+time_len].reshape(-1, dX.shape[-1])
                M_dX = self.projection(dX1).reshape(
                    N, -1, self.channels, self.hidden_size, self.hidden_size)
                out = torch.einsum('bcij,bcjk->bcik', out,
                                   self.dyadic_prod(M_dX))
            return out

    @ staticmethod
    def dyadic_prod(X: torch.tensor) -> torch.tensor:
        """compute cumulative product on matrix time series
            with dyadic partition, should have complexity in O(log(T))

        Args:
            X (torch.tensor): A batch of matrix time series, shape (N,T,m,m)

        Returns:
            torch.tensor: cumulative product on time dimension, shape (N,T,m,m)
        """
        N, T, C, m, m = X.shape
        max_level = int(np.ceil(np.log2(T)))
        I = torch.eye(m, device=X.device, dtype=X.dtype).reshape(
            1, 1, 1, m, m).repeat(N, 1, C, 1, 1)
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum('bcij,bcjk->bcik', X[:, 0], X[:, 1])
            # X = torch.bmm(X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)
        return X[:, 0]

    @ staticmethod
    def prod(X):
        M = []
        N, T, C, m, m = X.shape
        I = torch.eye(m, device=X.device).reshape(
            1, 1, m, m).repeat(N, C, 1, 1)
        M_X = I
        M.append(I.view(N, 1, C, m, m))
        for i in range(T):
            M_X = torch.einsum('bcij,bcjk->bcik', M_X, X[:, i])
            M.append(
                M_X.view(N, 1, C, m, m))
        return torch.cat(M, dim=1)
