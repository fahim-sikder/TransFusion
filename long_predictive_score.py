import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from torch import Tensor, nn
from tqdm import trange

from utils import extract_time


def long_predictive_score_metrics(ori_data, generated_data):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    ## Builde a post-hoc RNN predictive network
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    class PredictorTransformer(nn.Module):
        def __init__(self, in_features_dim, hidden_dim, seq_len, nhead=8):
            super().__init__()
            self.projection = nn.Linear(in_features_dim - 1, hidden_dim)
            self.pos_encoding = PositionalEncoding(
                d_model=hidden_dim, dropout=0, max_len=seq_len + 1
            )  # TODO: reduce seq+N
            encoder_block = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=hidden_dim, batch_first=False
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_block, num_layers=1
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.projection(x)  
            x = x.permute(
                1, 0, 2
            )  
            x = self.pos_encoding(x)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  
            y_hat_logit = self.fc(x)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat


    model = PredictorTransformer(
        in_features_dim=dim, hidden_dim=hidden_dim, seq_len=seq_len
    ).to(device)

    # Loss for the predictor
    p_loss = nn.L1Loss()
    # optimizer
    p_optimizer = torch.optim.Adam(model.parameters())

    batch_size = min((batch_size, len(generated_data)))

    # Training
    for itt in trange(iterations):
        p_optimizer.zero_grad()
        # Set mini-batch
        idx = torch.randperm(len(generated_data))
        train_idx = idx[:batch_size]

        # selection of  batch with pytorch approach
        X_mb = torch.index_select(generated_data[:, :-1, : (dim - 1)], 0, train_idx)
        Y_mb = torch.index_select(generated_data[:, 1:, (dim - 1)], 0, train_idx)
        Y_mb = Y_mb.reshape(batch_size, seq_len - 1, 1)

        X_mb = X_mb.to(device)
        Y_mb = Y_mb.to(device)

        y_pred_mb = model(X_mb)
        loss = p_loss(Y_mb, y_pred_mb)
        loss.backward()
        p_optimizer.step()



    ## Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = list(ori_data[i][:-1, : (dim - 1)] for i in train_idx)
    Y_mb = list(
        np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1])
        for i in train_idx
    )

    model = model.to("cpu")
    MAE_temp = 0
    for i in range(no):
        with torch.no_grad():

            pred_Y_curr = model(X_mb[i].unsqueeze(0)).squeeze(0)
            MAE_temp = MAE_temp + mean_absolute_error(
                Y_mb[i].detach(), pred_Y_curr.detach()
            )

    predictive_score = MAE_temp / no

    return predictive_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        d_model = d_model if (d_model % 2) == 0 else d_model + 1
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        seq_len, _, emb_dim = x.shape
        x = x + self.pe[:seq_len, :, :emb_dim]
        return self.dropout(x)