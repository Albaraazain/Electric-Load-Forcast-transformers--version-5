from dataclasses import dataclass
from typing import Optional

@dataclass
class InformerConfig:
    """Configuration for Informer model"""

    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.2
    activation: str = 'gelu'
    distil: bool = True

    # Data parameters
    input_features: int = 8  # Energy + time features
    input_window: int = 192  # 2 days of 15-min intervals
    prediction_window: int = 192  # 2 days ahead prediction
    stride: int = 1

    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    patience: int = 10  # Early stopping patience
    warmup_steps: int = 4000
    grad_clip_value: float = 1.0

    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Device configuration
    device: str = 'cuda'

    def __post_init__(self):
        """Validate configuration"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, "Split ratios must sum to 1"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.input_window > 0 and self.prediction_window > 0, "Window sizes must be positive"