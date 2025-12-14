"""
Model Utilities: Save and Load Trained PINN Models
"""

import torch
from pathlib import Path
from datetime import datetime


def save_model(model, history, filepath, hyperparams=None):
    """Save trained model with history and hyperparameters."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'hyperparams': hyperparams or {},
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath, model_class, device=None):
    """Load trained model from file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    hyperparams = checkpoint.get('hyperparams', {})
    hidden_layers = hyperparams.get('hidden_layers', 4)
    neurons_per_layer = hyperparams.get('neurons_per_layer', 64)
    
    model = model_class(hidden_layers, neurons_per_layer).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {filepath}")
    print(f"  Architecture: {hidden_layers} layers × {neurons_per_layer} neurons")
    
    return model, checkpoint.get('history', {}), hyperparams


def list_saved_models(directory="."):
    """List all saved .pt model files."""
    model_files = list(Path(directory).glob("*.pt"))
    if model_files:
        print(f"Found {len(model_files)} model(s):")
        for f in sorted(model_files):
            print(f"  - {f.name}")
    return model_files
