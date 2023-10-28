#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchsummary
import torch.nn.init as init

def init_weights(m):
    if isinstance(m,nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(m.bias.data)


class EncoderBlock(nn.Module):
    def __init__(self,input,n_filters,dropout_prob=0.3, max_pooling = True,padding="same"):
        super(EncoderBlock,self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Conv1d(input,n_filters,kernel_size=9,padding=padding)
        self.conv2 = nn.Conv1d(n_filters,n_filters,kernel_size=9,padding=padding)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.max_pooling = max_pooling
        self.max_pool = nn.MaxPool1d(2)
        self.batch_norm = nn.BatchNorm1d(n_filters)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv1.bias)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv2.bias)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        if self.max_pooling:
            x = self.max_pool(x)
        else:
            x = x
        skip_connection = x
        return x, skip_connection

class DecoderBlock(nn.Module):
    def __init__(self,input,n_filters,padding="same"):
        super(DecoderBlock,self).__init__()
        self.conv1 = nn.Conv1d(n_filters,n_filters,kernel_size=9,padding=padding)
        self.conv2 = nn.Conv1d(n_filters,n_filters,kernel_size=9,padding=padding)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.upsample = nn.ConvTranspose1d(input,n_filters,kernel_size=9,stride=1,padding=4)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv1.bias)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv2.bias)


    def forward(self,x,skip_connection):
        x = self.upsample(x)
        merge = torch.cat([x,skip_connection],dim=2)
        x = self.conv1(merge)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
    

class WaveLike(nn.Module):
    def __init__(self, n_filters=32, dropout_prob=0.3):
        super(WaveLike, self).__init__()
        self.n_filters = n_filters
        self.dropout_prob = dropout_prob
        self.encoder1 = EncoderBlock(1,n_filters,dropout_prob=0,max_pooling=True)
        self.encoder2 = EncoderBlock(n_filters,n_filters*2,dropout_prob=0,max_pooling=True)
        self.encoder3 = EncoderBlock(n_filters*2,n_filters*4,dropout_prob=0,max_pooling=True)
        self.encoder4 = EncoderBlock(n_filters*4,n_filters*8,dropout_prob=0.3,max_pooling=True)
        self.encoder5 = EncoderBlock(n_filters*8,n_filters*16,dropout_prob=0.3,max_pooling=False)
        self.decoder1 = DecoderBlock(n_filters*16,n_filters*8)
        self.decoder2 = DecoderBlock(n_filters*8,n_filters*4)
        self.decoder3 = DecoderBlock(n_filters*4,n_filters*2)
        self.decoder4 = DecoderBlock(n_filters*2,n_filters)

        self.conv = nn.Conv1d(n_filters,n_filters,kernel_size=9,padding="same")
        self.relu = nn.ReLU()
        self.last_conv = nn.Conv1d(n_filters,1,kernel_size=1,padding="same")

        init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv.bias)


    def forward(self,x):
        x1= self.encoder1(x)
    
        x2 = self.encoder2(x1[0])

        x3 = self.encoder3(x2[0])
        
        x4 = self.encoder4(x3[0])
        
        x5 = self.encoder5(x4[0])
        
        z6 = self.decoder1(x5[0],x4[1])
        
        z7 = self.decoder2(z6,x3[1])
        
        z8 = self.decoder3(z7,x2[1])
        
        z9 = self.decoder4(z8,x1[1])
        
        conv = self.conv(z9)
        conv = self.relu(conv)
        conv = self.last_conv(conv)
        
        return conv

#%%
model = WaveLike().to('cuda')
torchsummary.summary(model, (1, 1024))
# %%
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PARENT_FOLDER = "/home/tudor/Documents/9thSemester/WaveUNet-Bass/WaveUNet-Bass/Data"
SAMPLING_RATE = 48000
EPSPILON = 1e-8
# May need adjusting
WINDOW_SIZE = int(0.02 * SAMPLING_RATE)
STRIDE_LENGTH = int(0.01 * SAMPLING_RATE)
SCALER = MinMaxScaler()
#%%
import numpy as np
def create_dataset_npy(parent_folder):
    input_samples, target_samples = [],[]
    for fold in os.listdir(parent_folder):
        subfolder = os.path.join(parent_folder,fold)
        if os.path.isdir(subfolder):
            in_path = os.path.join(subfolder,"admm_reference.npy")
            target_path = os.path.join(subfolder,"admm_processed.npy")
            in_sg = np.load(in_path)
            target_sg = np.load(target_path)
            for i in range(len(in_sg)):
                input_sample = SCALER.fit_transform(in_sg[i].reshape(-1,1))
                target_sample = SCALER.fit_transform(target_sg[i].reshape(-1,1))
                input_samples.append(input_sample)
                target_samples.append(target_sample)
    return input_samples, target_samples 
# %%
input_samples_npy, target_samples_npy = create_dataset_npy(PARENT_FOLDER)
#%%
print(f"Input samples: {len(input_samples_npy)}, Shape of samples: {input_samples_npy[0].shape}, min and max values: {np.min(input_samples_npy[0])} and {np.max(input_samples_npy[0])}")
print(f"Target samples: {len(target_samples_npy)}, Shape of samples: {target_samples_npy[0].shape}, min and max values: {np.min(target_samples_npy[0])} and {np.max(target_samples_npy[0])}")
# %%
from torch.utils.data import Dataset, DataLoader, random_split

df = pd.DataFrame({"input":input_samples_npy,"target":target_samples_npy})
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
# %% Training functions
import matplotlib.pyplot as plt
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

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


model = WaveLike().to(device)
# Mean Squared Error Loss
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-6)
# %%
dataset = BassenhanceDataset(df)
train_dataset, val_dataset = dataset.split()

# %%
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
# %%
train_losses, val_losses = train(model, train_loader, valid_loader, optimizer, criterion, device, epochs=1000)
#%% SAVE FOR TRANSFER LEARNING
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # <-- this part
            'loss': loss,
            ...
            }, "like_niels_no_norm.pth")
            # %%
plot_losses_real_time(train_losses, val_losses)
# %%
len(val_dataset)
#%%
import matplotlib.pyplot as plt
import torch

model.eval()
with torch.no_grad():
    for i in range(100):
        predict = model(val_dataset[i][0].unsqueeze(0).to(device))
        predict = predict.squeeze(0).cpu().numpy()
        target = val_dataset[i][1].squeeze(0).cpu().numpy()
        
        # Create a new plot for each data point
        plt.figure()
        
        # Plot the predicted data
        plt.plot(predict, label="Predicted")
        
        # Plot the target data
        plt.plot(target, label="Target")
        
        # Add labels and legend
        # Show or save the plot as needed
        # For example, to display the plot:
    plt.show()

# %%
PARENT_FOLDER = "/home/tudor/Documents/9thSemester/WaveUNet-Bass/WaveUNet-Bass/Predict"
unseen_input,unseen_target = create_dataset_npy(PARENT_FOLDER)
#%%
unseen_input = torch.tensor(unseen_input, dtype=torch.float32).to(device)
unseen_target = torch.tensor(unseen_target, dtype=torch.float32).to(device)
#%%
unseen_input = unseen_input.transpose(1,2)
unseen_target = unseen_target.transpose(1,2)
# %%
import numpy as np
def compute_similarity(target, predict, weightage=[0.33, 0.33, 0.33]):
    try:
        # Time domain similarity
        ref_time = np.correlate(target, target, mode='full')
        inp_time = np.correlate(target, predict, mode='full')
        diff_time = np.abs(ref_time - inp_time)

        # Freq domain similarity
        ref_freq = np.correlate(np.fft.fft(target), np.fft.fft(target), mode='full')
        inp_freq = np.correlate(np.fft.fft(target), np.fft.fft(predict), mode='full')
        diff_freq = np.abs(ref_freq - inp_freq)

        # Power similarity
        ref_power = np.sum(target**2)
        inp_power = np.sum(predict**2)
        diff_power = np.abs(ref_power - inp_power)

        return float(weightage[0] * np.mean(diff_time) + weightage[1] * np.mean(diff_freq) + weightage[2] * np.mean(diff_power))
    except Exception as e:
        print(f"An error occurred: {e}")
        return None 
#%%
unseen_predictions = []
model.eval()
with torch.no_grad():
    for i in range(10):
        predict = model(unseen_input[i].unsqueeze(0).to(device))
        predict = predict.squeeze(0).cpu().numpy()
        target = unseen_target[i].squeeze(0).cpu().numpy()
        similarity = compute_similarity(target.flatten(),predict.flatten())
        print(similarity)
        unseen_predictions.append(predict)
# %%
import scipy as sp
import scipy.signal

class segmenter():
    def __init__(self,segment_size, overlap, window):
        self.segment_size = segment_size
        self.overlap = overlap
        self.hop_size = self.segment_size - self.overlap
        self.window = window


# %% COMPUTE LOSSES FOR
# MSE
# FOURIER LOSS
# ENERGY LOSS

