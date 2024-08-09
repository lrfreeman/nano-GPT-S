"""This module contains the configuration classes for a base instance of a decoder
transformer model that I will then port from an NLP implementation to a neural implementation"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """This class contains the configuration for the model"""
    d_model: int = 256
    d_head: int = 64
    d_mlp: int = d_model * 4
    n_blocks: int = 2 # 2 layers per block (Attention + MLP)
    n_heads: int = d_model // d_head
    n_ctx: int = 256
    init_range: float = 0.02 # std for initializing weights
    dropout: float = 0.1
    
@dataclass
class TransformerTrainingArgs():
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: Optional[str] = "base_transformer"
    wandb_name: Optional[str] = None