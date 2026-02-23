import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """LSTM for locomotion mode classification"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=2, 
                 num_classes=5, dropout=0.2, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        B, T, H = lstm_out.shape
        logits_flat = self.fc(lstm_out.reshape(-1, H))
        return logits_flat.reshape(B, T, -1)


