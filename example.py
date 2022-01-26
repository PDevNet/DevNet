from development.nn import development_layer, expm
from development.sp import sp
from development.hyperbolic import hyperbolic
from development.se import se
from development.so import so
import torch

# Create some data
batch, length, input_size = 1, 10, 2
hidden_size = 3
x = torch.rand(batch, length, input_size)

# specify the Lie algebra
param = so  # sp, hyperbolic, se,


# Development layer can directly apply on x like standard RNN model
# Specify development layer with the weigths

dev = development_layer(
    input_size=input_size, hidden_size=hidden_size, param=param, return_sequence=False)
out = dev(x)
