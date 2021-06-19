import torch
from torch import nn
from torch.nn.functional import interpolate
import random
import math

class Demucs(nn.Module):

    def __init__(self, audio_channels=1, num_layers=5, num_channels=64, kernel_size=8, stride=2, resample=2, bidirectional=True):
        super().__init__()
        self.audio_channels = audio_channels
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.bidirectional = bidirectional
        
        # Check if resample is not a valide value
        if resample == 1 or resample == 2 or resample == 4:
            self.resample = resample
        else:
            print('\nResample must be 1, 2 or 4. Default value is set to 1\n')
            self.resample = 1
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.lstm = nn.ModuleList()
        

        in_channels = audio_channels
        channels = num_channels

        for layer in range(self.num_layers):
            self.encoder.append(nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=self.kernel_size, stride=self.stride))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=1, stride=1))
            self.encoder.append(nn.GLU(dim=1))
            
            if layer > 0:
                self.decoder.insert(0, nn.ReLU())
            self.decoder.insert(0, nn.ConvTranspose1d(in_channels=channels, out_channels=in_channels, kernel_size=self.kernel_size, stride=self.stride))
            self.decoder.insert(0, nn.GLU(dim=1))
            self.decoder.insert(0, nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=1, stride=1))
               

            in_channels = channels
            channels = 2*channels
        
        self.lstm.append(nn.LSTM(bidirectional=self.bidirectional, num_layers=2, hidden_size=in_channels, input_size=in_channels))
        if self.bidirectional:
            self.lstm.append(nn.Linear(2*in_channels, in_channels)) # Resize BiLSTM output

    
    def pad_length(self, length):
        
        # Determinate the padding length for the input and the output to be of equal length
        final_length = length
        final_length = math.ceil(final_length * self.resample) # Length after the upsampling
        # Length after encoding network
        for idx in range(self.num_layers):
            final_length = math.ceil((final_length - self.kernel_size) / self.stride) + 1
            final_length = max(final_length, 1)
        # Length after decoding network
        for idx in range(self.num_layers):
            final_length = (final_length - 1) * self.stride + self.kernel_size
        final_length = int(math.ceil(final_length / self.resample)) # Length after the downsampling
        return int(final_length) - length


    def forward(self, x):

        # Normalize
        std = x.std(dim=-1, keepdim=True)
        x = x/std

        # Padding
        x_len = x.size()[-1]
        x = nn.functional.pad(x, pad=[0, self.pad_length(x_len)])

        # Upsampling
        x = interpolate(x, scale_factor=self.resample, mode='linear', align_corners=True)

        # Encoding
        encoder_outputs = []
        i = 0
        for layer in self.encoder:
            x = layer(x)
            # If last layer of encoder add the output to the skip network
            if i%4 == 3:
                encoder_outputs.append(x)
            i += 1
        
        # BiLSTM
        x = x.permute(2, 0, 1)
        x, _ = self.lstm[0](x)
        if self.bidirectional:
            x = self.lstm[1](x)
        x = x.permute(1, 2, 0)

        # Decoding
        i = 0
        for layer in self.decoder:
            # If first layer of decoder add the output of the coerrespondig encoder to the input
            if i%4 == 0:
                in_sum = encoder_outputs.pop(-1)
                x = x + in_sum
            x = layer(x)
            i +=1
        
        # Downsampling
        x = interpolate(x, scale_factor=1/self.resample, mode='linear', align_corners=True)
        
        # Eliminate padding
        x = x[..., :x_len]

        # Denomarlize
        return std * x


    def print_model(self):
        print('\n\nencoder:\n')
        for layer in self.encoder:
            print(layer)
        print('\n\ndecoder:\n')
        for layer in self.decoder:
            print(layer)
        print('\n\n')
        


def test():
    x = torch.randn(1,1,1000)
    print(x.size())

    demucs = Demucs(num_layers=4, resample=2)
    #demucs.print_model()
    demucs.forward(x)

if __name__ == '__main__':
    test()