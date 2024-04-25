import os
import pickle


import numpy as np
import torch

from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def normalize(data):
    
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    data = data / (max_val + 1e-7)
    
    data = data.astype(np.float32)
    
    return data
#################################################

class Sine_Pytorch(torch.utils.data.Dataset):
    
    def __init__(self, no_samples, seq_len, features):
        
        self.data = []
        
        for i in range(no_samples):
            
            temp = []
            
            for k in range(features):
                
                freq = np.random.uniform(0, 0.1)
                
                phase = np.random.uniform(0, 0.1)
                
                temp_data = [np.sin(freq*j + phase) for j in range(seq_len)]
                
                temp.append(temp_data)
                
            temp = np.transpose(np.asarray(temp))
            
            temp = (temp + 1) * 0.5
            
            self.data.append(temp)
        
        self.data = np.asarray(self.data, dtype = np.float32)
        
    def __len__(self):
        
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        
        return self.data[idx, :, :]
#################################################    

def data_preprocess(dataset_name):
    
    data_dir = f'data'
    
    if dataset_name == 'air':
        
        data = pd.read_csv(f'{data_dir}/AirQualityUCI.csv', delimiter= ';', decimal = ',')
        
        # Last 114 rows does not contain any values
        
        data = data.iloc[:-114, 2:15]
        
    elif dataset_name == 'energy':
        
        data = pd.read_csv(f'{data_dir}/energydata_complete.csv')
        
        data = data.iloc[:, 1:]
        
    elif dataset_name == 'stock':
        
        data = pd.read_csv(f'{data_dir}/GOOG.csv')
        
        data = data.iloc[:, 1:]
        
    return data


class MakeDATA(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        
        data = np.asarray(data, dtype= np.float32)
        
        data = data[::-1]

        norm_data = normalize(data)

        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)

        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
            
        self.samples = np.asarray(self.samples, dtype = np.float32)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    
def LoadData(dataset_name, seq_len):
    
    if dataset_name == 'sine':
        
        data = Sine_Pytorch(5000, seq_len, 5)
        
        train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 2021)
        
        print(f'Sine data loaded with sequence {seq_len}')
        
    else:
        
        data = data_preprocess(dataset_name)
        
        data = MakeDATA(data, seq_len)
        
        train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 2021)
        
        print(f'{dataset_name} data loaded with sequence {seq_len}')
        
    return train_data, test_data