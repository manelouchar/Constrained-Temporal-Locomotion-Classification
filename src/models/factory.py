from .lstm_frame import LSTMClassifier
from .cnn_lstm import CNNLSTMClassifier, CNNLSTMClassifierDeep


def create_cnn_lstm_model(model_type="standard", **kwargs):
    """Model factory"""
    if model_type == "standard":
        return CNNLSTMClassifier(**kwargs)
    elif model_type == "deep":
        return CNNLSTMClassifierDeep(**kwargs)
    elif model_type == "light":
        kwargs.update({
            'cnn_channels': [32, 64],
            'kernel_sizes': [5, 3],
            'hidden_size': 64,
            'num_layers': 1
        })
        return CNNLSTMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")   