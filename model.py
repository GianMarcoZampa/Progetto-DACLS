import torch
from torch import nn
from torch.nn.functional import interpolate

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
        
        # Upsampling
        x = interpolate(x, scale_factor=self.resample, mode='linear', align_corners=True)

        # Autoencoder calculations
        encoder_outputs = []
        i = 0
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        for layer in self.decoder:
            print(x.size())
            print(encoder_outputs[-1-i].size())
            print(layer)
            x = x +  encoder_outputs[-1-i][..., :x.shape[-1]]
            x = layer(x)
            i +=1
        
        # Downsampling
        x = interpolate(x, scale_factor=1/self.resample, mode='linear', align_corners=True)


    def print_model(self):
        print('\n\nencoder:\n')
        for layer in self.encoder:
            print(layer)
        print('\n\ndecoder:\n')
        for layer in self.decoder:
            print(layer)
        print('\n\n')
        


def test():

    x = torch.tensor([[range(256)]], dtype=torch.float32)
    '''
    print(a)
    a = interpolate(a, scale_factor=2, mode='linear', align_corners=True)
    print(a)
    a = interpolate(a, scale_factor=1/2, mode='linear', align_corners=True)
    print(a)
    '''

    demucs = Demucs(num_layers=2, resample=2)
    #demucs.print_model()
    demucs.forward(x)

if __name__ == '__main__':
    test()