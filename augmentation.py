import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import random
from filters import BP_filter
import matplotlib.pyplot as plt

class Random_shift(nn.Module):
# Randomly shifts the track
    def __init__(self, shift_max):
        super().__init__()
        self.shift_max = shift_max


    def forward(self, noisy_track, clean_track):
        shift = random.randint(0, self.shift_max)
        noisy_track = torch.roll(noisy_track, shift, dims=-1)
        clean_track = torch.roll(clean_track, shift, dims=-1)
        return noisy_track, clean_track


class Remix(nn.Module):
# Randomly shifts the noise in the track
    def forward(self, noisy_track, clean_track):
        noise = noisy_track - clean_track
        perm = torch.randperm(noise.size()[-1])
        noisy_track[-1][-1][:] = clean_track[-1][-1][:] + noise[-1][-1][perm]
        return noisy_track, clean_track


class Band_mask(nn.Module):
# Maskes bands of frequencies
    def __init__(self, sample_rate, min_freq, max_freq):
        super().__init__()
        self.sample_rate = sample_rate
        self.filter = BP_filter(min_freq, max_freq)


    def forward(self, noisy_track, clean_track):
        noisy_track = self.filter.forward(noisy_track)
        clean_track = self.filter.forward(clean_track)
        return noisy_track, clean_track



def test():
    x = torch.randn(1,1,1000)
    y = torch.randn(1,1,1000)
    t = np.linspace(0,1000, num=1000, dtype=np.float32)
    
    shift = Random_shift(10)
    remix = Remix()
    band_mask = Band_mask(48000, 0.1, 0.15)
    shifted_x, shifted_y = shift.forward(x, y)
    remixed_x, remixed_y = remix.forward(x, y)
    masked_x, masked_y = band_mask.forward(x, y)

    plt.figure()
    plt.title('x')
    plt.plot(t, x[-1][-1])
    plt.figure()
    plt.title('shifted x')
    plt.plot(t, shifted_x[-1][-1])
    plt.figure()
    plt.title('remixed x')
    plt.plot(t, remixed_x[-1][-1])
    plt.figure()
    plt.title('masked x')
    plt.plot(t, masked_x[-1][-1])
    plt.show()



if __name__ == '__main__':
    test()

