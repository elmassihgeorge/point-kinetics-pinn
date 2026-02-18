"""Model utilities — save / load trained PINN checkpoints."""

import torch
from pathlib import Path
from datetime import datetime


def save_model(model, history, filepath, hyperparams=None):
    """Save trained model with history and hyperparameters."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "hyperparams": hyperparams or {},
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, filepath)
    print(f"  Model saved: {filepath}")


def load_model(filepath, model_class, device=None):
    """Load a trained model from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    hp = checkpoint.get("hyperparams", {})
    hidden = hp.get("hidden_layers", 4)
    neurons = hp.get("neurons_per_layer", 64)

    model = model_class(hidden, neurons).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Model loaded: {filepath}  ({hidden} layers x {neurons} neurons)")
    return model, checkpoint.get("history", {}), hp


def list_saved_models(directory="."):
    """List all .pt checkpoint files in *directory*."""
    files = sorted(Path(directory).glob("*.pt"))
    if files:
        print(f"Found {len(files)} model(s):")
        for f in files:
            print(f"  - {f.name}")
    return files
