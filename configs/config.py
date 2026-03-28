from dataclasses import dataclass


@dataclass
class ModelConfig:
    # shared
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    vocab_size: int = 256
    dropout: float = 0.1

    # BDH
    mlp_internal_dim_multiplier: int = 128

    # Transformer (FIX)
    ffn_dim: int = 1024  # 4 * n_embd


@dataclass
class TrainConfig:
    block_size: int = 128
    batch_size: int = 8

    # debugging phase
    max_iters: int = 10
    log_freq: int = 1

    # training
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 42