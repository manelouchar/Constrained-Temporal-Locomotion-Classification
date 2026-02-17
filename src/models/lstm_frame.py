import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """LSTM simple pour classification de gait modes"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 5, 
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        Args:
            input_size: nombre de features (27 dans votre cas)
            hidden_size: taille de la couche cachée LSTM
            num_layers: nombre de couches LSTM
            num_classes: nombre de classes (5)
            dropout: taux de dropout
            bidirectional: LSTM bidirectionnel ou non
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Couche de classification
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
    
    def forward(self, x):
        """Frame-wise classification (sortie à chaque timestep)"""
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        
        # Appliquer FC à CHAQUE timestep
        batch_size, seq_len, hidden_dim = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(-1, hidden_dim)
        logits_flat = self.fc(lstm_out_flat)
        
        return logits_flat.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 5)
    
    def forward_sequence(self, x):
        """
        Forward avec output à chaque timestep (pour sequence labeling)
        
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Apply FC à chaque timestep
        # Reshape: (batch * seq_len, hidden_size * num_directions)
        batch_size, seq_len, hidden_dim = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(-1, hidden_dim)
        
        # FC: (batch * seq_len, num_classes)
        logits_flat = self.fc(lstm_out_flat)
        
        # Reshape back: (batch, seq_len, num_classes)
        logits = logits_flat.reshape(batch_size, seq_len, -1)
        
        return logits