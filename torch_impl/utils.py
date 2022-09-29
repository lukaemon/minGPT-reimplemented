import random
import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple
from enum import Enum

from flax import struct

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ModelConfig:
    vocab_size: int
    context_window: int
    n_embd: int = None
    n_head: int = None
    n_layer: int = None
    model_type: str = None
    attn_pdrop: float = 0.1
    recid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    

    def __post_init__(self):
        self.config_set = {
            # names follow the huggingface naming conventions
            # GPT-1
            "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
            # GPT-2 configs
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
            # Gophers
            "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
            # (there are a number more...)
            # I made these tiny models up
            "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
            "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
            "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
        }

        params_given = all([self.n_embd, self.n_head, self.n_layer])
        type_given = self.model_type is not None
        assert (
            type_given ^ params_given
        ), f"Supported model_type: {list(self.config_set.keys())}, or provide customized params."

        if type_given:
            preset = self.config_set[self.model_type]
            self.n_embd = preset["n_embd"]
            self.n_head = preset["n_head"]
            self.n_layer = preset["n_layer"]


TrainerCallbackEvent = Enum("TrainerCallbackEvent", "on_train_batch_end")


@dataclass
class TrainConfig:
    device: str = "auto"

    # dataloader
    num_workers: int = 0
    batch_size: int = 64
    sequence_len: int = 6

    # training
    max_iters: int = None
    learning_rate: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0
