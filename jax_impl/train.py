import jax
from jax import tree_util

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
    logits = jnp.where(
        (labels == ignore_index)[..., None], -nn.log_softmax(logits), logits
    )  # ref 2, omg, price for jit
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)  # ref 1

    # batch average loss, torch cross entropy loss default redution is mean as well
    return loss.mean()


# filter = labels != ignore_index  # filter out -1 padding pairs
# labels = labels[filter]
# logits = logits[filter]
# NonConcreteBooleanIndexError: Array boolean indices must be concrete; got ShapedArray(bool[32,11])
# See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError


def create_weight_decay_mask(params):  # FIXME: You can do better, this is crazy
    """
    Don't apply weight decay to bias, layernorm and embeddings
    Apply weight decay to the rest of params

    Achieve this in few steps:
    - flatten the param tree, get the structure of the tree
    - traversing param tree to get full name list for each leaf param
    - apply truth value funciton to names to decide whether its lr should be decayed
    - build a tree with param tree structure and the truth value list, this is the mask

    Later, pass the mask to optax optimizer
    https://optax.readthedocs.io/en/latest/api.html#adamw
    """

    def traversal(tree):
        result = []

        def walk(node, buffer):
            if isinstance(node, jnp.DeviceArray):
                result.append(".".join(buffer))
            else:
                for key in node.keys():
                    walk(node[key], buffer + [key.lower()])

        walk(tree, [])
        return result

    def is_weight_decay(name):
        no_decay = ["bias", "layernorm", "embedding"]

        out = True
        for nono in no_decay:
            out &= nono not in name

        return out

    ps, structure = tree_util.tree_flatten(
        params, is_leaf=lambda node: isinstance(node, jnp.DeviceArray)
    )

    params_name = traversal(params)
    assert len(ps) == len(params_name), "traversal error, len didn't match"

    flat_mask = list(map(is_weight_decay, params_name))
    mask = tree_util.tree_unflatten(structure, flat_mask)

    return mask


def create_train_state(rng, cfg: Config):
    rng_param, rng_dropout, rng = jax.random.split(rng, 3)
    init_rngs = {"params": rng_param, "dropout": rng_dropout}

    model = GPT(cfg)

    # FIXME: dataset specific hack ... you can do better
    block_size = cfg.sequence_len * 2 - 1
    input_shape = (cfg.batch_size, block_size)
    params = model.init(init_rngs, jnp.zeros(input_shape, jnp.int32))

    weight_decay_mask = create_weight_decay_mask(params)
    optimizer = optax.chain(
        optax.clip(cfg.grad_norm_clip),
        optax.adamw(
            cfg.learning_rate,
            cfg.b1,
            cfg.b2,
            weight_decay=cfg.weight_decay,
            mask=weight_decay_mask,
        ),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

# all bets are on being able to jax.jit this, otherwise, what's the purpose of using JAX
def train_step(state: train_state.TrainState, batch, cfg: Config, dropout_rng):
    def compute_loss(params):
        x, y = batch
        logits = GPT(cfg).apply(params, x, rngs={"dropout": dropout_rng})
        loss = loss_fn(logits, y, ignore_index=-1)

        return loss, logits

    def compute_accuracy(logits, labels):
        pred = jnp.argmax(logits, axis=-1)

        # still need to take care of -1 cas
        acc = (pred == labels).mean(where=labels != -1)

        return acc

    _, labels = batch

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, "acc": compute_accuracy(logits, labels)}

    return state, metrics


# reference
# 1. cross entropy loss https://github.com/deepmind/optax/blob/90b17710a061a5f4c7b060f4c7715f2d539cb39f/optax/_src/loss.py#L175#L207
# 2. cross entropy loss and normalizing note
# cross entropy:
# H(P, Q) = - expectation_p(x)(log(q(x)))
#         = - sum(p(x)log(q(x))) # In multi-classification case, p(x) = 1 for correct case, the rest is 0
#         = - log(q(x)) # predicted prob at correct class logit
# q = logits
# q(x) = softmax(logits) = exp(x) / sum(exp(logits))
# let i = correct index
# -log(q(i)) = -log(exp(i) / sum(exp(logits)))
#            = sum(exp(logits)) - i # this is how optax implement cross_entropy loss
# https://github.com/deepmind/optax/blob/master/optax/_src/loss.py#L175#L207
#
# normalization of logits for correct loss computation:
# logits = jnp.where((labels == ignore_index)[..., None], -nn.log_softmax(logits), logits)
#
# data looks like this
# x =      [0   2  5  3  3  1  0  1  2  3  3] # shape= (11, )
# labels = [-1 -1 -1 -1 -1  0  1  2  3  3  5] # shape= (11, )
#
# For all lables = -1, you don't want to compute loss since they are just padding position.
# To compute loss of multi-class cross entropy, you got logits of shape (11, num_class)
# since loss = -log(q(i)) = sum(exp(logits)) - i
# in padding possitions, i=-1, this is invalid position, the loss contribution should be 0
# hence sum(exp(logits)) should equal to i, put it in equaiton
# sum(exp(logits)) = i = -1
# that's why we need to pre-normalize logits with -logsoftmax at location where label = -1
# sum(exp(-logsoftmax(logits))) = -sum(softmax(logits)) = -1
#
# However, you can't use numpy truth value index in jax.jit, hence the usage of jnp.where
# (labels == ignore_index)[..., None] do this for proper shape of broadcasting by add a dim at the end.
