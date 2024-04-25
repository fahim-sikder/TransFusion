import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide_torch, extract_time, batch_generator
import torch
from torch import nn, Tensor
from tqdm import trange
from tqdm import tqdm
import math


def long_discriminative_score_metrics(ori_data, generated_data):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape    

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
        
    # Network parameters
    hidden_dim = max((int(dim/2),1))
    iterations = 2000
    batch_size = 128

    class Discriminator_GRU(nn.Module):
        def __init__(self, dim, hidden_dim):
            super(Discriminator_GRU, self).__init__()
            self.p_cell = nn.GRU(dim, hidden_dim, batch_first = True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, d_last_states = self.p_cell(x)
            y_hat_logit = self.fc(d_last_states)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat
        
    
    class Discriminator_Trans(nn.Module):
        def __init__(self, num_tokens, feature_dim, hidden_dim=3, nhead=8, num_layers=1):
            super(Discriminator_Trans, self).__init__()
            self.projection = nn.Linear(feature_dim, hidden_dim)
            self.positional_encoding = PositionalEncoding(d_model=hidden_dim, dropout=0, max_len=num_tokens+10)
            encoder_block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=False)
            self.transformer_encoder = nn.TransformerEncoder(encoder_block, num_layers=num_layers)
            self.fc = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            
            x = self.projection(x)
            x = x.permute(1, 0, 2)  
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  
            x = torch.mean(x, dim=1, keepdim=True).permute(1, 0, 2)  
            y_hat_logit = self.fc(x)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat


    # model
    discriminator = Discriminator_Trans(num_tokens=seq_len, feature_dim=dim, hidden_dim=8).to(device)
    
    # optimizer
    d_optimizer = torch.optim.Adam(discriminator.parameters())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat = \
    train_test_divide_torch(ori_data, generated_data, ori_time, generated_time)


    loss = nn.functional.binary_cross_entropy_with_logits

    # Training step
    for itt in tqdm(range(iterations)):
        d_optimizer.zero_grad()
            
        # Batch setting
        no = len(train_x)
        idx = torch.randperm(no)
        train_idx = idx[:batch_size] 
        X_mb = torch.index_select(train_x, 0, train_idx)
        no = len(train_x_hat)
        idx = torch.randperm(no)
        train_idx = idx[:batch_size] 
        X_hat_mb = torch.index_select(train_x_hat, 0, train_idx)    
        
        X_mb = X_mb.to(device)
        X_hat_mb = X_hat_mb.to(device)

        # model inference        
        y_logit_real, y_pred_real = discriminator(X_mb)
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)

        # loss calculation 
        d_loss_real = loss(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = loss(y_logit_fake, torch.zeros_like(y_logit_fake))  
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        d_optimizer.step()   
        
    test_x = test_x.to(device)
    
    test_x_hat = test_x_hat.to(device)
    
    with torch.no_grad():
  
        _, y_pred_real_curr = discriminator(test_x)
        y_pred_real_curr = y_pred_real_curr.squeeze()
        _, y_pred_fake_curr = discriminator(test_x_hat)
        y_pred_fake_curr = y_pred_fake_curr.squeeze()

    y_pred_final = torch.cat((y_pred_real_curr, y_pred_fake_curr), dim = 0)
    y_label_final = torch.cat((torch.ones([len(y_pred_real_curr),]), torch.zeros([len(y_pred_fake_curr),])), dim = 0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final.cpu().numpy(), (y_pred_final.cpu().numpy()>0.5))
    discriminative_score = np.abs(0.5-acc)

    return discriminative_score


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)