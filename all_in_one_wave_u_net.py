#%%
import os
import torchaudio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PARENT_FOLDER = "/mnt/c/Users/Tudor/Documents/yt-dlp/"
SAMPLING_RATE = 48000
EPSPILON = 1e-8
# May need adjusting
WINDOW_SIZE = int(0.02 * SAMPLING_RATE)
STRIDE_LENGTH = int(0.01 * SAMPLING_RATE)
SCALER = MinMaxScaler()

#%% IF YOU HAVE WAV FILES
def create_dataset_wav(parent_folder):
    input_samples, target_samples = [],[]
    for fold in os.listdir(parent_folder):
        sub_folder_path = os.path.join(PARENT_FOLDER,fold)
        if os.path.isdir(sub_folder_path):
            in_path = os.path.join(sub_folder_path,"admm_reference.wav")
            target_path = os.path.join(sub_folder_path,"admm_processed.wav")
            in_sg, in_sr = torchaudio.load(in_path)
            target_sg, target_sr = torchaudio.load(target_path)
            print(f"Input {in_path} and Target {target_path} have {in_sg.shape[1]} samples")
            num_ch, num_fr = in_sg.shape
            assert in_sr == target_sr, f"Input {in_sr}Hz and Target {target_sr} need to have the same Sampling Rate {SAMPLING_RATE}"
            assert num_ch == 1 , f"Only support for mono, you have {num_ch} channels"
            assert in_path != target_path , f'Input path is {in_path} and target path is {target_path}'

            for ch in range(num_ch):
                input_data, target_data = in_sg[ch,:], target_sg[ch,:]
                for i in range(0,len(input_data) - WINDOW_SIZE,STRIDE_LENGTH):
                    input_sample = SCALER.fit_transform(input_data[i:i+WINDOW_SIZE].reshape(-1,1))
                    target_sample = SCALER.fit_transform(target_data[i:i+WINDOW_SIZE].reshape(-1,1))
                    input_samples.append(input_sample)
                    target_samples.append(target_sample)
        if fold == "AZB_12":
            break
    return input_samples, target_samples

#%% IF YOU HAVE NPY FILES

import numpy as np
def create_dataset_npy(parent_folder):
    input_samples, target_samples = [],[]
    for fold in os.listdir(parent_folder):
        sub_folder_path = os.path.join(PARENT_FOLDER,fold)
        if os.path.isdir(sub_folder_path):
            in_path = os.path.join(sub_folder_path,"admm_reference.npy")
            target_path = os.path.join(sub_folder_path,"admm_processed.npy")
            in_sg = np.load(in_path)
            target_sg = np.load(target_path)
            for i in range(len(in_sg)):
                input_sample = SCALER.fit_transform(in_sg[i].reshape(-1,1))
                target_sample = SCALER.fit_transform(target_sg[i].reshape(-1,1))
                input_samples.append(input_sample)
                target_samples.append(target_sample)
        if fold == "AZB_12":
            break
    return input_samples, target_samples 

#%%
input_samples_npy, target_samples_npy = create_dataset_npy(PARENT_FOLDER)
#%%
input_samples_wav, target_samples_wav = create_dataset_wav(PARENT_FOLDER)
#%%
print(len(input_samples_npy),len(target_samples_npy))
#%%
print(len(input_samples_wav),len(target_samples_wav))
#%% Check if normalization is correct
print("NPY")
print(min(input_samples_npy[0]),max(input_samples_npy[0]))
print(min(target_samples_npy[0]),max(target_samples_npy[0]))
print("WAV")
print(min(input_samples_wav[0]),max(input_samples_wav[0]))
print(min(target_samples_wav[0]),max(target_samples_wav[0]))
# %% Plot samples
import matplotlib.pyplot as plt

for i in range(30, 51):
    # Create a new figure and axes for each iteration
    fig, ax = plt.subplots()
    
    # Plot the 'input' and 'target' on separate subplots
    ax.plot(input_samples[i], label='Input')
    
    # You can customize the plot for the 'target' as needed
    ax.plot(target_samples[i], label='Target')
    
    ax.set_title(f"Plot for Sample {i + 1}")
    ax.legend()
    
    # Show or save the plot as needed
    plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=9, stride=1, padding="same"):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.main(x)
        return self.dropout(x)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=9, stride=1, padding="same"):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.main(x)
    

class Model(nn.Module):
    def __init__(self, n_layers=8, channels_interval=16):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, kernel_size=3, stride=1,
                      padding="same"),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1+self.channels_interval, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # Initialize the weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        tmp = []
        o = x
        # Up Sample
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            o = F.max_pool1d(o, kernel_size=2, stride=2)

        o = self.middle(o)

        for i in range(self.n_layers):
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            o = torch.cat((o, tmp[self.n_layers - i - 1]), dim=1)
            o = self.decoder[i](o)
        o = torch.cat((o, x), dim=1)
        o = self.out(o)
        return o
# %%
obj = Model()
# %%
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
#%% DATASET IF NP ARRAY
class BassenhanceDatasetNPY(Dataset):
    def __init__(self, input_samples, target_samples):
        self.input_samples = input_samples
        self.target_samples = target_samples
    def __len__(self):
        return len(self.input_samples)
    
    def __getitem__(self, idx):
        input = self.input_samples[idx]
        target = self.target_samples[idx]

        input = torch.tensor(input, dtype=torch.float32).T
        target = torch.tensor(target, dtype=torch.float32).T

        return input, target
    
# %% DATASET IF DF

df = pd.DataFrame({"input":input_samples,"target":target_samples})
class BassenhanceDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.input = df["input"]
        self.target = df["target"]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]

        input = torch.tensor(input, dtype=torch.float32).T
        target = torch.tensor(target, dtype=torch.float32).T

        return input, target
    
    
    def get_loader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)
    
    def transpose(self, data):
        return data.transpose(1,2)
    
    def get_loader_transpose(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn, drop_last=True)
    
    def split(self, train_size=0.8, shuffle=True):
        return torch.utils.data.random_split(self, [int(len(self) * train_size), len(self) - int(len(self) * train_size)], generator=torch.Generator().manual_seed(42))
    
# TRAIN FUNCTIONS
#%%
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss = loss / (torch.linalg.vector_norm(target, ord=2) + EPSPILON)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            loss = loss / (torch.linalg.vector_norm(target, ord=2) + EPSPILON)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        save_state(model, epoch + 1)
        if early_stopping(val_losses, patience=50):
            print("Early Stopping")
            break
    return train_losses, val_losses

def plot_losses_real_time(train_losses, val_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience:
        return False
    else:
        return val_losses[-1] > val_losses[-2] > val_losses[-3]
    


# Save state every 10 epochs
def save_state(model, epoch, path = "models"):
    if epoch % 10 == 0:
        state = { "epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict() }
        torch.save(state, os.path.join(path, f"model_{epoch}.pth"))
        print("Saved model")

def load_state(model, optimizer, path = "models"):
    state = torch.load(path)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]
    return model, optimizer, epoch
#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


model = Model().to(device)
# Mean Squared Error Loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
#%%
df = pd.DataFrame({"input":input_samples,"target":target_samples})
dataset = BassenhanceDataset(df)
train_dataset, val_dataset = dataset.split()

#%%
print(len(train_dataset),len(val_dataset),len(dataset))
#%%
print(train_dataset[0][0].shape,train_dataset[0][1].shape)
#%%
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
valid_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=2)
#%%
import matplotlib.pyplot as plt
for i,(input,target) in enumerate(val_dataset):
    if i == 2:
        input, target = input.squeeze().cpu().numpy(), target.squeeze().cpu().numpy()
        fig, axs = plt.subplots(figsize=(20, 10))
        axs.plot(input, label="Input")
        axs.plot(target, label="Target")
        axs.legend()
        plt.show()
        break
#%%
train_losses, val_losses = train(model, train_loader, valid_loader, optimizer, criterion, device, epochs=200)
# %%
torch.save(model.state_dict(), "test.pth")
#%%
import matplotlib.pyplot as plt
plot_losses_real_time(train_losses, val_losses)

#%% Predict with the model
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

def predict(model, input, target, device):
    model.eval()
    with torch.no_grad():
        input, target = input.to(device), target.to(device)
        output = model(input)
        output = output.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        input = input.squeeze().cpu().numpy()
        return input, target, output
    
def plot_predictions(input, target, output):
    fig, axs = plt.subplots(figsize=(20, 10))
    axs.plot(input, label="Input")
    axs.plot(target, label="Target")
    axs.plot(output, label="Output")
    axs.legend()
    plt.show()
#%%
print(val_dataset[0][0].shape,val_dataset[0][1].shape)
#%%
model.load_state_dict(torch.load("models/model_40.pth"))
#%%
def show_predictions(model, val_dataset, device, max_samples=10):
    for i,(input,target) in enumerate(val_dataset):
        input, target, output = predict(model, input.unsqueeze(0), target.unsqueeze(0), device)
        plot_predictions(input, target, output)
        if i == max_samples:
            break
#%%
a = torch.arange(9, dtype=torch.float) - 4
b = torch.linalg.vector_norm(a, ord=2)
c = torch.norm(a, p=2)
print(b,c)
# %%
show_predictions(model, val_dataset, device, max_samples=3)
# %%
