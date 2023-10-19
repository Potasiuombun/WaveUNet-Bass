#%%
import os
import torchaudio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PARENT_FOLDER = "/home/tudor/Documents/9thSemester/Wave-U-Net/Data"
SAMPLING_RATE = 48000
# May need adjusting
WINDOW_SIZE = int(0.02 * SAMPLING_RATE)
STRIDE_LENGTH = int(0.01 * SAMPLING_RATE)
SCALER = MinMaxScaler()

#%%
def create_dataset(parent_folder):
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
    return input_samples, target_samples


#%%
input_samples, target_samples = create_dataset(PARENT_FOLDER)

# %%
df = pd.DataFrame({"input":input_samples,"target":target_samples})
#%%
df.to_pickle("dataset.pkl")
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.main(x)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
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
    def __init__(self,n_layers = 2, channels_interval=4):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        print(encoder_in_channels_list)
        print(encoder_out_channels_list)


        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )
        print(self.encoder)

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, kernel_size=3, stride=1,
                      padding="same"),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        print(self.middle)

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        print(decoder_in_channels_list)
        print(decoder_out_channels_list)
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )
        print(self.decoder)

        self.out = nn.Sequential(
            nn.Conv1d(1+self.channels_interval, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        print(self.out)

    def forward(self, x):
        tmp = []
        o = x
        print(o.shape)
        # Up Sample
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            o = F.max_pool1d(o, kernel_size=2, stride=2)
            print(o.shape)

        o = self.middle(o)

        for i in range(self.n_layers):
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            o = torch.cat((o, tmp[self.n_layers - i - 1]), dim=1)
            o = self.decoder[i](o)
            print(o.shape)
        o = torch.cat((o, x), dim=1)
        o = self.out(o)
        print(o.shape)
        return o
# %%
obj = Model()
# %%
import torch.optim as optim
# %%
import pandas as pd
import pickle

df = pd.read_pickle("dataset.pkl")
# %%
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
for i in range(30, 51):
    # Create a new figure and axes for each iteration
    fig, ax = plt.subplots()
    
    # Plot the 'input' and 'target' on separate subplots
    ax.plot(df["input"][i], label='Input')
    
    # You can customize the plot for the 'target' as needed
    ax.plot(df["target"][i], label='Target')
    
    ax.set_title(f"Plot for Sample {i + 1}")
    ax.legend()
    
    # Show or save the plot as needed
    plt.show()

# %%
from torch.utils.data import Dataset, DataLoader, random_split

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

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
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
    return train_losses, val_losses

def plot_losses_real_time(train_losses, val_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


model = Model().to(device)
# Mean Squared Error divided by target_Frame L2 Norm
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = BassenhanceDataset(df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
#%%
train_losses, val_losses = train(model, train_loader, train_loader, optimizer, criterion, device, epochs=10)
# %%
