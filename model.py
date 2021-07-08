import torch
from torch import nn
from torch.nn.functional import interpolate
from torchinfo import summary
import math

class Demucs(nn.Module):

    def __init__(self, audio_channels=1, num_layers=5, num_channels=64, kernel_size=8, stride=2, resample=2, LSTM=True, bidirectional=True):
        super().__init__()
        self.audio_channels = audio_channels
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.LSTM = LSTM
        self.bidirectional = bidirectional
        
        # Check if resample is not a valide value
        if resample == 1 or resample == 2 or resample == 4:
            self.resample = resample
        else:
            print('\nResample must be 1, 2 or 4. Default value is set to 1\n')
            self.resample = 1
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        

        in_channels = audio_channels
        channels = num_channels

        self.bn1 = nn.BatchNorm1d(1)

        for layer in range(self.num_layers):
            self.encoder.append(nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=self.kernel_size, stride=self.stride, padding=3))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=1, stride=1))
            self.encoder.append(nn.GLU(dim=1))
            
            if layer > 0:
                self.decoder.insert(0, nn.ReLU())
            self.decoder.insert(0, nn.ConvTranspose1d(in_channels=channels, out_channels=in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=3))
            self.decoder.insert(0, nn.GLU(dim=1))
            self.decoder.insert(0, nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=1, stride=1))
               

            in_channels = channels
            channels = 2*channels
        
        if self.LSTM:
            self.lstm = nn.ModuleList()
            self.lstm.append(nn.LSTM(bidirectional=self.bidirectional, num_layers=2, hidden_size=in_channels, input_size=in_channels))
            if self.bidirectional:
                self.lstm.append(nn.Linear(2*in_channels, in_channels)) # Resize BiLSTM output


    def forward(self, x):

        # Normalize
        #x = self.bn1(x)

        # Upsampling
        x = interpolate(x, scale_factor=self.resample, mode='linear', align_corners=True, recompute_scale_factor=True)
        #print(f'After upsample: {x.shape}')

        # Encoding
        encoder_outputs = []
        i = 0
        for layer in self.encoder:
            x = layer(x)
            #print(f'Encoder: {x.shape}')
            # If last layer of encoder add the output to the skip network
            if i%4 == 3:
                encoder_outputs.append(x)
            i += 1

        # BiLSTM
        if self.LSTM:
            x = x.permute(2, 0, 1)
            #print(f'Input LSTM 1: {x.shape}')
            x, _ = self.lstm[0](x)
            #print(f'Output LSTM 1: {x.shape}')
            if self.bidirectional:
                x = self.lstm[1](x)
                #print(f'Output LSTM 2: {x.shape}')
            x = x.permute(1, 2, 0)
            #print(f'After permutation: {x.shape}')

        # Decoding
        i = 0
        for layer in self.decoder:
            # If first layer of decoder add the output of the coerrespondig encoder to the input
            if i%4 == 0:
                in_sum = encoder_outputs.pop(-1)
                x = x + in_sum
            x = layer(x)
            #print(f'Decoder: {x.shape}')
            i +=1
        
        # Downsampling
        x = interpolate(x, scale_factor=1/self.resample, mode='linear', align_corners=True, recompute_scale_factor=True)
        #print(f'After downsample: {x.shape}')

        # Denomarlize
        #return std * x
        #return x * max_x
        return x


    def print_model(self):
        print('\n\nencoder:\n')
        for layer in self.encoder:
            print(layer)
        print('\n\ndecoder:\n')
        for layer in self.decoder:
            print(layer)
        print('\n\n')
        


def test():
    x = torch.randn(1,1,48000)
    y = torch.randn(1,1,48000)
    
    demucs = Demucs()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))

    optimizer.zero_grad()
    out = demucs.forward(x)
    loss = loss_func(out, y)
    loss.backward()
    optimizer.step()

    print(loss.item())

    #summary(demucs, input_size=(1, 1, 48000))


if __name__ == '__main__':
    test()