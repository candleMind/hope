import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class SupUnsupFusion(nn.Module):
    """
    Fusion model that combines supervised and unsupervised predictions.

    This model takes predictions from both supervised and unsupervised methods
    and fuses them using a feedforward neural network to produce final predictions.
    """

    def __init__(self, sup_size=8, unsup_size=8, label_size=8, hidden_size=None):
        """
        Initialize the fusion model.

        Args:
            sup_size: Dimension of supervised predictions (number of supervised features)
            unsup_size: Dimension of unsupervised predictions (number of unsupervised features)
            label_size: Number of output labels
            hidden_size: Size of hidden layer (defaults to sup_size + unsup_size if None)
        """
        super().__init__()
        if hidden_size is None:
            hidden_size = sup_size + unsup_size  # Default to input size

        self.fusion = nn.Sequential(
            nn.Linear(sup_size + unsup_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, label_size),
        )

    def forward(self, x_sup, x_unsup):
        """
        Forward pass through the fusion model.

        Args:
            x_sup: Supervised predictions tensor (batch_size, sup_size)
            x_unsup: Unsupervised predictions tensor (batch_size, unsup_size)

        Returns:
            Fused predictions tensor (batch_size, label_size)
        """
        x = torch.cat([x_sup, x_unsup], dim=-1)
        x = self.fusion(x)
        return x


def get_model(sup_size, unsup_size, label_size, hidden_size=None):
    """
    Create and return a fusion model.

    Args:
        sup_size: Dimension of supervised predictions
        unsup_size: Dimension of unsupervised predictions
        label_size: Number of output labels
        hidden_size: Size of hidden layer (optional)

    Returns:
        SupUnsupFusion model instance on device
    """
    model = SupUnsupFusion(
        sup_size=sup_size,
        unsup_size=unsup_size,
        label_size=label_size,
        hidden_size=hidden_size,
    ).to(device)

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    return model


def load_checkpoint(
    checkpoint_path, sup_size, unsup_size, label_size, hidden_size=None
):
    """
    Load a trained fusion model from checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint
        sup_size: Dimension of supervised predictions
        unsup_size: Dimension of unsupervised predictions
        label_size: Number of output labels
        hidden_size: Size of hidden layer (optional)

    Returns:
        Loaded SupUnsupFusion model instance on device
    """
    model = SupUnsupFusion(
        sup_size=sup_size,
        unsup_size=unsup_size,
        label_size=label_size,
        hidden_size=hidden_size,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    print(f"Model loaded from {checkpoint_path}")
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    return model
