import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class BP_filter(nn.Module):
    def __init__(self, min_freq, max_freq, width=None):
        super().__init__()
        self.low_filter = LP_filter(min_freq, width)
        self.high_filter = LP_filter(max_freq, width)

        

    def forward(self, track):
        low_track = self.low_filter.forward(track)
        high_track = self.high_filter.forward(track)
        return track - high_track + low_track
    
    def show_filter(self):
        plt.figure(0)
        plt.title('Low filter')
        plt.plot(self.low_filter.x, self.low_filter.weights)
        plt.figure(1)
        plt.title('High filter')
        plt.plot(self.high_filter.x, self.high_filter.weights)
        plt.show()


class LP_filter(nn.Module):
    def __init__(self, cutoff_freq, width=None):
        super().__init__()

        if width is None:
            width = int(2 / cutoff_freq)
        self.width = width

        self.x = np.linspace(-width, width+1, num=2*width+1, dtype=np.float32)
        sinc = torch.from_numpy(np.sinc(2*cutoff_freq*self.x))
        window = torch.hamming_window(2*width+1, periodic=False)

        self.weights = 2 * cutoff_freq * sinc * window

    
    def forward(self, track):
        track = nn.functional.conv1d(track, weight=self.weights.unsqueeze(-2).unsqueeze(-2), padding=self.width)
        return track


    def show_filter(self):
        plt.figure(0)
        plt.title('Filter')
        plt.plot(self.x, self.weights)
        plt.show()


def test():
    lp = LP_filter(0.6)
    bp = BP_filter(0.2, 0.4)
    lp.show_filter()
    bp.show_filter()


if __name__ == '__main__':
    test()


