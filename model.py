import torch
from torch import nn


class Demucs(nn.Module):
    def __init__(self, num_layers=2, audio_channels=1, num_channels=64, kernel_size=8, stride=4, resample=1):
        super().__init__()
        self.num_layers = num_layers
        self.audio_channels = audio_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if resample == 1 or resample == 2 or resample == 4:
            self.resample = resample
        else:
            print('\nResample must be 1, 2 or 4. Default value is set to 1\n')
            self.resample = 1
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        in_channels = audio_channels
        channels = num_channels

        for layer in range(self.num_layers):
            self.encoder.append(nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=self.kernel_size, stride=self.stride))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=1, stride=1))
            self.encoder.append(nn.GLU(dim=1))
            
            if layer > 0:
                self.decoder.insert(0, nn.ReLU())
            self.decoder.insert(0, nn.ConvTranspose1d(in_channels=channels, out_channels=in_channels, kernel_size=1, stride=1))
            self.decoder.insert(0, nn.GLU(dim=1))
            self.decoder.insert(0, nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=self.kernel_size, stride=self.stride))
               

            in_channels = channels
            channels = 2*channels
    

    def forward(self, x):
        layer_outputs = []
        for layer in self.encoder:
            x = layer(x)
            layer_outputs.append(x)
        for layer in self.decoder:
            x += layer_outputs[-1-i]
            x = layer(x)


    def print_model(self):
        print('\n\nencoder:\n')
        for layer in self.encoder:
            print(layer)
        print('\n\ndecoder:\n')
        for layer in self.decoder:
            print(layer)
        print('\n\n')
        


def test():
    demucs = Demucs(num_layers=1, resample=1)
    demucs.print_model()

if __name__ == '__main__':
    test()