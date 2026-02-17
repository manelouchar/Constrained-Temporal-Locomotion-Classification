import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM Hybrid Architecture pour classification de modes de locomotion
    - CNN 1D: extraction de features locales temporelles
    - LSTM: capture des dépendances temporelles longues
    """
    
    def __init__(
        self, 
        input_size: int,
        num_classes: int = 5,
        # CNN params
        cnn_channels: list = [64, 128, 128],
        kernel_sizes: list = [5, 5, 3],
        # LSTM params
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: nombre de features IMU (27)
            num_classes: nombre de classes (5)
            cnn_channels: canaux pour chaque couche CNN
            kernel_sizes: tailles de kernel pour chaque couche CNN
            hidden_size: taille LSTM
            num_layers: nombre de couches LSTM
            dropout: taux de dropout
            bidirectional: LSTM bidirectionnel
        """
        super(CNNLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ========================================
        # CNN 1D Feature Extractor
        # ========================================
        cnn_layers = []
        in_channels = input_size
        
        for out_channels, kernel_size in zip(cnn_channels, kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # dropout plus léger dans CNN
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_channels[-1]
        
        # ========================================
        # LSTM Temporal Context
        # ========================================
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # ========================================
        # Classifier Head
        # ========================================
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
    
    def forward(self, x):
        """
        Forward pass avec sortie frame-wise
        
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        batch_size, seq_len, input_size = x.shape
        
        # CNN attend (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, cnn_channels[-1], seq_len)
        
        # Retour à (batch, seq_len, channels) pour LSTM
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, hidden_size * num_directions)
        lstm_out = self.dropout(lstm_out)
        
        # Classification frame-wise
        # Reshape pour appliquer FC à chaque timestep
        lstm_out_flat = lstm_out.reshape(-1, self.hidden_size * self.num_directions)
        logits_flat = self.fc(lstm_out_flat)
        
        # Reshape back
        logits = logits_flat.reshape(batch_size, seq_len, -1)
        
        return logits


class CNNLSTMClassifierDeep(nn.Module):
    """
    Version plus profonde avec Max Pooling pour réduire la dimension temporelle
    Utile pour des séquences très longues
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 5,
        # CNN params
        cnn_channels: list = [64, 128, 256],
        kernel_sizes: list = [7, 5, 3],
        pool_sizes: list = [2, 2, 1],  # pooling après chaque conv
        # LSTM params
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(CNNLSTMClassifierDeep, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pool_sizes = pool_sizes
        
        # CNN blocks with pooling
        cnn_layers = []
        in_channels = input_size
        
        for out_channels, kernel_size, pool_size in zip(cnn_channels, kernel_sizes, pool_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity(),
                nn.Dropout(dropout * 0.5)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_channels[-1]
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        
        # Upsampling pour revenir à la résolution originale
        total_pool = 1
        for p in pool_sizes:
            total_pool *= p
        self.upsample_factor = total_pool
    
    def forward(self, x):
        """
        Forward avec upsampling pour garder seq_len original
        """
        batch_size, seq_len, input_size = x.shape
        
        # CNN
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.dropout(lstm_out)
        
        # Upsampling si nécessaire (interpolation linéaire)
        if self.upsample_factor > 1:
            lstm_out = lstm_out.transpose(1, 2)  # (batch, hidden, reduced_seq)
            lstm_out = nn.functional.interpolate(
                lstm_out, 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            )
            lstm_out = lstm_out.transpose(1, 2)  # (batch, seq_len, hidden)
        
        # Classification
        lstm_out_flat = lstm_out.reshape(-1, self.hidden_size * self.num_directions)
        logits_flat = self.fc(lstm_out_flat)
        logits = logits_flat.reshape(batch_size, seq_len, -1)
        
        return logits


# ========================================
# Model Factory
# ========================================
def create_cnn_lstm_model(model_type: str = "standard", **kwargs):
    """
    Factory pour créer différentes variantes CNN-LSTM
    
    Args:
        model_type: "standard", "deep", ou "light"
        **kwargs: paramètres du modèle
    
    Returns:
        model: CNN-LSTM model
    """
    if model_type == "standard":
        return CNNLSTMClassifier(**kwargs)
    
    elif model_type == "deep":
        return CNNLSTMClassifierDeep(**kwargs)
    
    elif model_type == "light":
        # Version légère pour expérimentation rapide
        kwargs.update({
            'cnn_channels': [32, 64],
            'kernel_sizes': [5, 3],
            'hidden_size': 64,
            'num_layers': 1
        })
        return CNNLSTMClassifier(**kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ========================================
# Test du modèle
# ========================================
if __name__ == "__main__":
    # Test dimensions
    batch_size = 8
    seq_len = 150  # 1.5s @ 100Hz
    input_size = 27
    num_classes = 5
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("="*60)
    print("Testing CNN-LSTM Models")
    print("="*60)
    
    # Standard model
    model_std = create_cnn_lstm_model(
        model_type="standard",
        input_size=input_size,
        num_classes=num_classes,
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 5, 3],
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    
    print(f"\n1. Standard CNN-LSTM")
    print(f"   Parameters: {sum(p.numel() for p in model_std.parameters()):,}")
    
    logits_std = model_std(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {logits_std.shape}")
    assert logits_std.shape == (batch_size, seq_len, num_classes), "Wrong output shape!"
    print("   ✓ Shape OK")
    
    # Deep model
    model_deep = create_cnn_lstm_model(
        model_type="deep",
        input_size=input_size,
        num_classes=num_classes,
        cnn_channels=[64, 128, 256],
        kernel_sizes=[7, 5, 3],
        pool_sizes=[2, 2, 1],
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    
    print(f"\n2. Deep CNN-LSTM (with pooling)")
    print(f"   Parameters: {sum(p.numel() for p in model_deep.parameters()):,}")
    
    logits_deep = model_deep(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {logits_deep.shape}")
    assert logits_deep.shape == (batch_size, seq_len, num_classes), "Wrong output shape!"
    print("   ✓ Shape OK")
    
    # Light model
    model_light = create_cnn_lstm_model(
        model_type="light",
        input_size=input_size,
        num_classes=num_classes
    )
    
    print(f"\n3. Light CNN-LSTM")
    print(f"   Parameters: {sum(p.numel() for p in model_light.parameters()):,}")
    
    logits_light = model_light(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {logits_light.shape}")
    assert logits_light.shape == (batch_size, seq_len, num_classes), "Wrong output shape!"
    print("   ✓ Shape OK")
    
    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print("="*60)