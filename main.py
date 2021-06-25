import os
from loss import Multi_STFT_loss
from data import Audio_dataset, parse_data
from evaluation import evaluate_model
import torch
import model



def simulation_selector(num, batch_size):
    if num == 1:
        print("\nSimulation parameters n 1\n")
        L = 5 # Number of layers
        H = 64 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 2 # Layer stride
        U = 2 # Resampling factor

        demucs = model.Demucs(audio_channels=batch_size, num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=True)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    elif num == 2:
        print("\nSimulation parameters n 2\n")
        L = 5 # Number of layers
        H = 64 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 4 # Layer stride
        U = 4 # Resampling factor

        demucs = model.Demucs(audio_channels=batch_size, num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=False)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))

    elif num == 3:
        print("\nSimulation parameters n 3\n")
        L = 5 # Number of layers
        H = 48 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 4 # Layer stride
        U = 4 # Resampling factor

        demucs = model.Demucs(audio_channels=batch_size, num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=False)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    else:
        print("\nDefault simulation parameters\n")
        demucs = model.Demucs(audio_channels=batch_size)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    return demucs, optimizer


    
def train(dir):
    # Simulation 1 not causal
    # Simulation 2 causal (H=64 number of hidden channels)
    # Simulation 3 causal (H=48 number of hidden channels)
    simulation = 1
    epochs = 1
    batch_size = 1
    loss_func = Multi_STFT_loss()
    demucs, optimizer = simulation_selector(simulation, batch_size)

    #parse_data(dir+'_noisy.csv', os.path.join(dir, 'noisy'))

    train_dataset = Audio_dataset(dir+'_noisy.csv', dir)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demucs = demucs.to(device)
    demucs.train()


    for epoch in range(1, epochs+1):

        epoch_loss = 0.0

        for noisy_data, clean_data in train_dataloader:
            
            # Adjust Tensor Size to 3
            if len(noisy_data.size()) == 2:
                noisy_data = torch.unsqueeze(noisy_data, 0)
                clean_data = torch.unsqueeze(clean_data, 0)

            # Send data to device: cuda or cpu
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # forward pass
            outputs = demucs.forward(noisy_data)

            # compute loss
            loss = loss_func(outputs, clean_data)

            # backward propagation
            loss.backward()

            # optimization (update weights)
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        epoch_mean_loss = epoch_loss/len(train_dataloader)

        print(f'Epoch {epoch}/{epochs} - Loss: {epoch_mean_loss:.3f}')
    

if __name__ == '__main__':
    dataset_path = os.path.join('dataset', 'test')
    train(dataset_path)