import os
from pickle import dump, load
from pesq import pesq
from torch.nn.functional import interpolate
from loss import Multi_STFT_loss
from data import Audio_dataset, Audio_dataset_v1, parse_data
import torch
from torch import nn
import time
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

        demucs = Demucs(num_layers=L, num_channels=H, kernel_size=K, stride=S, resample=U, LSTM=False, bidirectional=True)
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


    
def train(simulation, dataset, epochs, batch_size=1, save_path='model.pt'):

    #loss_func = Multi_STFT_loss()
    loss_func = nn.MSELoss()
    demucs, optimizer = simulation_selector(simulation, batch_size)

    #parse_data(dataset+'_meta_file.csv', os.path.join('dataset', dataset, 'noisy'))
    
    #train_dataset = Audio_dataset('dataset_'+dataset+'.pt', 48000)
    train_dataset = Audio_dataset_v1(dataset+'_meta_file.csv', os.path.join('dataset', dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demucs = demucs.to(device)
    demucs.train()

    min_mean_loss = None

    # Training
    for epoch in range(1, epochs+1):
        epoch_time = time.time()
        epoch_loss = 0.0

        for noisy_data, clean_data in train_dataloader:
            
            ts = time.time()

            # Normalize
            max_noisy = torch.max(abs(noisy_data))
            noisy_data = noisy_data / max_noisy

            max_clean = torch.max(abs(clean_data))
            clean_data = clean_data / max_clean

            # Send data to device: cuda or cpu
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)

            #ts = time.time()
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            #print(f'Zero gradients time: {time.time()-ts}')

            #ts = time.time()
            # Forward pass
            outputs = demucs.forward(noisy_data)
            #print(f'Forward time: {time.time()-ts}')

            #ts = time.time()
            # Compute loss
            loss = loss_func(outputs, clean_data)
            #print(f'Loss time: {time.time()-ts}')

            #ts = time.time()
            # Backward propagation
            loss.backward()
            #print(f'Backward time: {time.time()-ts}')

            #ts = time.time()
            # Optimization (update weights)
            optimizer.step()
            #print(f'Optimization time: {time.time()-ts}')

            # Accumulate loss
            epoch_loss += loss.item()
            print(epoch_loss)
            print(f'One batch time: {time.time()-ts}')

        epoch_mean_loss = epoch_loss/len(train_dataloader)
        print(f'Epoch {epoch}/{epochs} - Loss: {epoch_mean_loss:.3f}')

        if min_mean_loss == None or epoch_mean_loss < min_mean_loss:
            # Save the model
            torch.save(demucs, save_path)

            min_mean_loss = epoch_mean_loss
        
        print(f'Epoch time: {time.time()-epoch_time}')



def evaluate(model_path, dataset, batch_size=1):

    # Loading trained model and sending to device
    demucs = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demucs = demucs.to(device)
    demucs.eval()

    #parse_data(dataset+'_meta_file.csv', os.path.join('dataset', dataset, 'noisy'))

    #train_dataset = Audio_dataset('dataset_'+dataset+'.pt', 48000)
    test_dataset = Audio_dataset_v1(dataset+'_meta_file.csv', os.path.join('dataset', dataset), cut=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Evaluation
    pesq_value = 0

    for noisy_data, clean_data in test_dataloader:
        ts = time.time()

        # Normalize
        max_noisy = torch.max(abs(noisy_data))
        noisy_data = noisy_data / max_noisy

        max_clean = torch.max(abs(clean_data))
        clean_data = clean_data / max_clean


        # Send data to device: cuda or cpu
        noisy_data = noisy_data.to(device)
        clean_data = clean_data.to(device)

        # Forward
        y_pred = demucs.forward(noisy_data)

        # Downsampling from 48000 to 16000
        clean_data = interpolate(clean_data, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)
        y_pred = interpolate(y_pred, scale_factor=1/3, mode='linear', align_corners=True, recompute_scale_factor=True)
        
        print(clean_data.shape)
        clean_data = clean_data[-1][-1].detach().numpy()
        y_pred = y_pred[-1][-1].detach().numpy()
        print(clean_data.shape)
        
        # PESQ evaluation
        pesq_value += pesq(16000, clean_data, y_pred, 'wb')
        print(pesq_value)
        print(f'One batch time: {time.time()-ts}')

    pesq_value /= len(test_dataloader)

    return pesq_value



if __name__ == '__main__':
    simulation = 1
    model_path = os.path.join('models', 'trained_model.pt')
    dataset = 'test'
    
    #train(simulation=simulation, dataset=dataset, epochs=1, batch_size=16)

    pesq_value = evaluate(model_path='model.pt', dataset=dataset)

    print(pesq_value)