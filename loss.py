import torch
import torch.nn as nn


class SC_loss(nn.Module):
    def forward(self, y, y_pred):
        # y clean signal
        # y_pred predicted signal

        # Return the Spectral Convergence loss with the Frobenius norm
        return torch.norm(y - y_pred, p="fro") / torch.norm(y, p="fro")


class Mag_loss(nn.Module):
    def forward(self, y, y_pred):
        # y clean signal
        # y_pred predicted signal

        # Return the L1 loss with the log of the Tensors
        return torch.functional.l1_loss(torch.log(y), torch.log(y_pred))


class STFT_loss(nn.Module):
    def __init__(self, fft_size=1024, hop_size=120, win_length=600):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length

        self.sc_loss = SC_loss()
        self.mag_loss = Mag_loss()

    def forward(self, y, y_pred):
        # y clean signal
        # y_pred predicted signal

        # Return the STFT loss, sum of SC and Mag losses using the Short time fourier transform
        y = torch.stft(y, self.fft_size, hop_lenght=self.hop_size, win_lenght=self.win_length)
        y_pred = torch.stft(y_pred, self.fft_size, hop_lenght=self.hop_size, win_lenght=self.win_length)
        
        sc_loss_value = self.sc_loss(y, y_pred)
        mag_loss_value = self.mag_loss(y, y_pred)

        return sc_loss_value + mag_loss_value


class Multi_STFT_loss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[50, 120, 240], win_lengths=[240, 600, 1200]):
        super().__init__()
        self.stft_losses = torch.nn.ModuleList()
        
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFT_loss(fs, hs, wl))


    def forward(self, y, y_pred):
        # y clean signal
        # y_pred predicted signal

        # Return the Multiresolution STFT loss
        stft_loss = 0.0
        
        for f in self.stft_losses:
            stft_loss_value = f(y, y_pred)
            stft_loss += stft_loss_value

        return stft_loss



def test():
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [50, 120, 240]
    win_lengths = [240, 600, 1200]
    for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
        print(fs, hs, wl)
    
    m_stft = Multi_STFT_loss()


if __name__ == '__main__':
    test()