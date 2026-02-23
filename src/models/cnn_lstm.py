import torch
import torch.nn as nn



class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM hybrid architecture"""
    
    def __init__(self, input_size, num_classes=5,
                 cnn_channels=[64, 128, 128], kernel_sizes=[5, 5, 3],
                 hidden_size=256, num_layers=2, dropout=0.2, 
                 bidirectional=True):
        super(CNNLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        cnn_layers, in_ch = [], input_size
        for out_ch, ks in zip(cnn_channels, kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_channels[-1]

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        B, T, _ = x.shape
        cnn_out = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.dropout(lstm_out)
        logits_flat = self.fc(lstm_out.reshape(-1, self.hidden_size * self.num_directions))
        return logits_flat.reshape(B, T, -1)


class CNNLSTMClassifierDeep(nn.Module):
    """Deep CNN-LSTM with pooling"""
    
    def __init__(self, input_size, num_classes=5,
                 cnn_channels=[64, 128, 256], kernel_sizes=[7, 5, 3],
                 pool_sizes=[2, 2, 1], hidden_size=256, num_layers=2,
                 dropout=0.2, bidirectional=True):
        super(CNNLSTMClassifierDeep, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pool_sizes = pool_sizes

        cnn_layers, in_ch = [], input_size
        for out_ch, ks, ps in zip(cnn_channels, kernel_sizes, pool_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.MaxPool1d(ps) if ps > 1 else nn.Identity(),
                nn.Dropout(dropout * 0.5)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_channels[-1]

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

        total_pool = 1
        for p in pool_sizes:
            total_pool *= p
        self.upsample_factor = total_pool

    def forward(self, x):
        B, T, _ = x.shape
        cnn_out = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.dropout(lstm_out)

        if self.upsample_factor > 1:
            lstm_out = nn.functional.interpolate(
                lstm_out.transpose(1, 2), size=T,
                mode='linear', align_corners=False
            ).transpose(1, 2)

        logits_flat = self.fc(lstm_out.reshape(-1, self.hidden_size * self.num_directions))
        return logits_flat.reshape(B, T, -1)
