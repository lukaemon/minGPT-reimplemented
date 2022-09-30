from flax import struct

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


# Because I use static_argname in jax.jit
# May not need to use special struc.dataclass here for the jit to work
# Could just use native python dataclass or ml_collections.ConfigDict
# Previous config overhaul is wasted haha. 
# 
# Also, I don't even need to use static_argname
# Just parital(train_step, cfg=cfg, ...) to wrap it up would work as well.
# This is how Flax wmt example did it. https://github.com/google/flax/blob/a3b4cad524cd7da4f044e10700f5af51b4d29c30/examples/wmt/train.py#L528

