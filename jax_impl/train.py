import jax
import jax.numpy as jnp
import flax.linen as nn

import optax
from flax.training import train_state

from jax_impl.model import GPT
from jax_impl.config import Config

# raw input, get the sense of -1 paddings 
# and the reason for crazy pre normalizing to ignore loss of padding positions
# [0 2 5 3 3 1 0 1 2 3 3]
# [-1 -1 -1 -1 -1  0  1  2  3  3  5]
def loss_fn(logits, labels, ignore_index=-1):
    logits = jnp.where((labels == ignore_index)[..., None], -nn.log_softmax(logits), logits) # ref 2, omg, price for jit
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)  # ref 1

    # batch average loss, torch cross entropy loss default redution is mean as well
    return loss.mean()

# filter = labels != ignore_index  # filter out -1 padding pairs
# labels = labels[filter]
# logits = logits[filter]
# NonConcreteBooleanIndexError: Array boolean indices must be concrete; got ShapedArray(bool[32,11])
# See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError

def create_train_state(rng, cfg: Config):
    rng_param, rng_dropout, rng = jax.random.split(rng, 3)
    init_rngs = {"params": rng_param, "dropout": rng_dropout}

    model = GPT(cfg)

    block_size = cfg.sequence_len * 2 - 1  # FIXME: dataset specific hack ... you can do better
    input_shape = (cfg.batch_size, block_size)
    params = model.init(init_rngs, jnp.zeros(input_shape, jnp.int32))
    tx = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(state: train_state.TrainState, batch, cfg: Config, dropout_rng):
    def compute_loss(params):
        x, y = batch
        logits = GPT(cfg).apply(params, x, rngs={"dropout": dropout_rng})
        loss = loss_fn(logits, y, ignore_index=-1)

        return loss, logits
    
    def compute_accuracy(logits, labels):  
        pred = jnp.argmax(logits, axis=-1)
        acc = (pred == labels).mean(where=labels != -1) # still need to take care of -1 case

        return acc

    x, labels = batch 

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = {
        'loss': loss,
        'acc': compute_accuracy(logits, labels)
        }

    return state, metrics


# reference
# 1. cross entropy loss https://github.com/deepmind/optax/blob/90b17710a061a5f4c7b060f4c7715f2d539cb39f/optax/_src/loss.py#L175#L207
# 2. TODO: cross entropy loss and normalizing note
