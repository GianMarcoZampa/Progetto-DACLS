import os
from pesq import pesq
from loss import Multi_STFT_loss
from data import Audio_dataset, parse_data
import torch
from model import Demucs



def simulation_selector(num, batch_size):
    # Simulation 1 not causal
    # Simulation 2 causal (H=64 number of hidden channels)
    # Simulation 3 causal (H=48 number of hidden channels)

    if num == 1:
        print("\nSimulation parameters n 1\n")
        L = 5 # Number of layers
        H = 64 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 2 # Layer stride
        U = 2 # Resampling factor

        demucs = Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=True)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    elif num == 2:
        print("\nSimulation parameters n 2\n")
        L = 5 # Number of layers
        H = 64 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 4 # Layer stride
        U = 4 # Resampling factor

        demucs = Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=False)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))

    elif num == 3:
        print("\nSimulation parameters n 3\n")
        L = 5 # Number of layers
        H = 48 # Number of hidden channels
        K = 8 # Layer kernel size
        S = 4 # Layer stride
        U = 4 # Resampling factor

        demucs = Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, bidirectional=False)
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    else:
        print("\nDefault simulation parameters\n")
        demucs = Demucs()
        optimizer = torch.optim.Adam(demucs.parameters(), lr=3e-4, betas=(0.9, 0.9999))
    
    return demucs, optimizer


    
def train(simulation, dir, epochs, batch_size=1, save_path=''):

    loss_func = Multi_STFT_loss()
    demucs, optimizer = simulation_selector(simulation, batch_size)

    #parse_data(dir+'_noisy.csv', os.path.join(dir, 'noisy'))
    
    train_dataset = Audio_dataset(dir+'_noisy.csv', dir)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demucs = demucs.to(device)
    demucs.train()

    # Training
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

            # Forward pass
            outputs = demucs.forward(noisy_data)

            # Compute loss
            loss = loss_func(outputs, clean_data)

            # Backward propagation
            loss.backward()

            # Optimization (update weights)
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        epoch_mean_loss = epoch_loss/len(train_dataloader)

        print(f'Epoch {epoch}/{epochs} - Loss: {epoch_mean_loss:.3f}')

    # Save the model
    torch.save(demucs, save_path)


def evaluate(model_path, dir, batch_size=1):

    # Loading the trained model
    demucs = torch.load(model_path)
    
    #parse_data(dir+'_noisy.csv', os.path.join(dir, 'noisy'))

    test_dataset = Audio_dataset(dir+'_noisy.csv', dir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Evaluation
    pesq_value = 0

    for noisy_data, clean_data in test_dataloader:
        if len(noisy_data.size()) == 2:
                noisy_data = torch.unsqueeze(noisy_data, 0)
                clean_data = torch.unsqueeze(clean_data, 0)

        y_pred = demucs.forward(noisy_data)
        pesq_value += pesq(test_dataset.sample_rate, clean_data, y_pred, 'wb')

    pesq_value /= test_dataset.__len__()

    return pesq_value



if __name__ == '__main__':
    simulation = 1
    model_path = ''
    train_dataset_path = os.path.join('dataset', 'test')
    test_dataset_path = os.path.join('dataset', 'test')

    train(simulation=simulation, dir=train_dataset_path, epochs=1)

    pesq = evaluate(model_path=model_path, dir=test_dataset_path)