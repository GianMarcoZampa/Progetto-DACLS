import torch
from torch import nn
import model

# Model parameters

L = 8 # Number of layers
H = 4 # Number of hidden channels
K = 8 # Layer kernel size
S = 4 # Layer stride
U = 2 # Resampling factor

demucs = model.Demucs()

a = torch.ones(3, dtype=torch.int)
print(a)
    

