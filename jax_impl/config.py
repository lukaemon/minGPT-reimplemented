from flax import struct


@struct.dataclass
class Config:
    # model config
    context_window: int = 128
    n_embd: int = 192
    n_head: int = 6
    n_layer: int = 6
    dropout_prob: float = 0.1

    # data config
    vocab_size: int = 64  # used to init dataset
    sequence_len: int = 64  # used to init dataset

    # train config
    batch_size: int = 64
    learning_rate: float = 5e-4
    n_epoch = 96

    # optimizer
    b1 = 0.9
    b2 = 0.95
    weight_decay = 0.1
    grad_norm_clip = 1.0


config_set = {
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

# Because I use static_argname in jax.jit
# May not need to use special struc.dataclass here for the jit to work
# Could just use native python dataclass or ml_collections.ConfigDict
# Previous config overhaul is wasted haha.
#
# Also, I don't even need to use static_argname
# Just parital(train_step, cfg=cfg, ...) to wrap it up would work as well.
# This is how Flax wmt example did it. https://github.com/google/flax/blob/a3b4cad524cd7da4f044e10700f5af51b4d29c30/examples/wmt/train.py#L528
