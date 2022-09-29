import jax
import jax.numpy as jnp 

import optax
from flax.training import train_state

from jax_impl.model import GPT
from torch_impl.utils import TrainConfig, ModelConfig

def loss_fn(logits, labels, ignore_index=-1):
    filter = labels != ignore_index # filter out -1 padding pairs
    labels = labels[filter]
    logits = logits[filter]

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) # ref 1

    return loss.mean() # batch average loss, torch cross entropy loss default redution is mean as well

def create_train_state(rng, model_config:ModelConfig, train_config: TrainConfig):
    rng_param, rng_dropout, rng = jax.random.split(rng, 3)
    init_rngs = {'params': rng_param, 'dropout': rng_dropout}

    model = GPT(model_config)
    
    input_shape = (train_config.batch_size, train_config.sequence_len)
    params = model.init(init_rngs, jnp.zeros(input_shape, jnp.int32))
    tx = optax.adamw(train_config.learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(rng, state:train_state.TrainState, batch, model_config):
    def compute_loss(params, rng):
        rng_dropout, rng = jax.random.split(rng)
        rngs = {'dropout': rng_dropout}
        
        x, y = batch
        logits = GPT(model_config).apply(params, x, rngs=rngs)
        loss = loss_fn(logits, y, ignore_index=-1)

        return loss
    grad_fn = jax.grad(compute_loss)
    grads = grad_fn(state.params, rng)

    state = state.apply_gradients(grads=grads)

    return state
    

# reference
# 1. cross entropy loss https://github.com/deepmind/optax/blob/90b17710a061a5f4c7b060f4c7715f2d539cb39f/optax/_src/loss.py#L175#L207
# 