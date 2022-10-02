from functools import partial
import jax.numpy as jnp
import jax

from jax_impl.model import GPT
from jax_impl.config import Config


def generate(cfg: Config, params, x, num_token):
    """greedy argmax
    x: (b, t)
    num_token: how many new token to generate
    """
    context = x

    infer_fn = jax.jit(partial(GPT(cfg).apply, params, training=False))

    for _ in range(num_token):
        # sliding window when the sequence is longer than context_window
        # context's max shape is (b, context_window)
        context = context[:, -cfg.context_window :]
        logits = infer_fn(context)

        # take the last token's logits, make the prediction, and adjust shape for next cat op
        pred = jnp.argmax(logits[:, -1, :], axis=-1)[..., None]  # (b, 1)

        context = jnp.concatenate([context, pred], axis=1)  # (b, t+1)

    return context
