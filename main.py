from data import Dataset
import random
import torch
from torch import nn
import model



def simulation_selector(num):
    if num == 1:
        print("\nSimulation parameters n 1\n")
        L = 5 # Number of layers
        H = 64 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 2 # Layer stride
        U = 2 # Resampling factor

        demucs = model.Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=True)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    elif num == 2:
        print("\nSimulation parameters n 2\n")
        L = 5 # Number of layers
        H = 64 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 4 # Layer stride
        U = 4 # Resampling factor

        demucs = model.Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=False)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))

    elif num == 3:
        print("\nSimulation parameters n 3\n")
        L = 5 # Number of layers
        H = 48 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 4 # Layer stride
        U = 4 # Resampling factor

        demucs = model.Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=False)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    else:
        print("\nDefault simulation parameters\n")
        demucs = model.Demucs()
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    return demucs, optimizer


def train():
    # Simulation 1 not causal
    # Simulation 2 causal (H=64 number of hidden channels)
    # Simulation 3 causal (H=48 number of hidden channels)
    simulation = 1
    epochs = 1
    loss_func = nn.MSELoss()
    demucs, optimizer = simulation_selector(simulation)
    
    if torch.cuda.is_available():
        demucs.cuda()
    
    
    train_dataset = Dataset('dataset/test')
    train_noisy, train_clean, sample_rate = train_dataset.get_data()


    print(train_noisy[0].size(), train_clean[0].size(), sample_rate)


    for epoch in range(epochs):
        train_clean_pred = demucs.forward(train_noisy)
        loss = loss_func(train_clean_pred, train_clean)
        loss.backward()
        print(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)%10 == 0:
            print(f'Epoch {epoch+1}, Loss {loss.item():.3f}')
   
    

if __name__ == '__main__':
    train()