from flax import struct

preset = {
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

@struct.dataclass
class Config:
    # model config
    context_window: int  = 64
    n_embd: int = 48
    n_head: int = 3
    n_layer: int = 3
    dropout_prob: float = 0.1


    # data config
    vocab_size: int = 8 # used to init dataset
    sequence_len: int = 6 # used to init dataset
    
    # train config
    batch_size: int  = 64
    learning_rate: float = 5e-4
    n_epoch = 12
    
    # optimizer
    b1 = 0.9
    b2 = 0.95
    weight_decay = 0.1
    grad_norm_clip = 1.0
