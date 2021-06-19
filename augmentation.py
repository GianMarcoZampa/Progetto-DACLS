import torch
import torch.nn as nn
import random

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
    def forward(self):
        pass



def test():
    x = torch.randn(1,1,100)
    y = torch.randn(1,1,100)
    
    shift = Random_shift(10)
    remix = Remix()
    shifted_x, shifted_y = shift.forward(x, y)
    remixed_x, remixed_y = remix.forward(x, y)



if __name__ == '__main__':
    test()

