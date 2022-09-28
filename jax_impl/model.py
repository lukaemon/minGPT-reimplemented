import math
import functools

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

import flax
import flax.linen as nn

import numpy as np

from general.utils import ModelConfig


class Attention(nn.Module):
    '''multihead causal self-attention'''
    config: ModelConfig

    @nn.compact
    def __call__(self, x, training=True):
        b, t, c = x.shape

        # [q, k, v], each (b,t,n_embd), pool qkv transform as one huge linear transform
        qkv = nn.Dense(
            features=3 * self.config.n_embd, # hence 3xn_embd output
            kernel_init=nn.initializers.normal(stddev=0.02) # bias init default zeros
            )(x).split(3, axis=-1) # then split
        
        # multi-head split, hc: head channel
        q, k, v = tree_map(
            lambda x: x.reshape(b, t, self.config.n_head, c // self.config.n_head).transpose((0, 2, 1, 3)),
            qkv,
        ) # (b,h,t,hc)

        att = q @ k.transpose(0, 1, 3, 2) / math.sqrt(c)  # (b,h,t,t)

        # apply causal mask
        mask = jnp.tril(jnp.ones((t, t)))
        att = jnp.where(mask==0, -jnp.inf, att)

        # compute weights
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(self.config.attn_pdrop)(att, deterministic=not training)

        # weighted sum
        output = att @ v  # (b,h,t,hc)

        # multihead concat, (b,t,h,hc) -> (b,t,c)
        output = output.transpose((0, 2, 1, 3)).reshape(b, t, c)

        # output dense layer
        output = nn.Dense(  # ref 3
            self.config.n_embd, 
            kernel_init=nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.config.n_layer))
            )(output)

        return output


class Block(nn.Module):
    '''encoder block'''
    config: ModelConfig

    @nn.compact
    def __call__(self, x, training=True):
        attn_out = nn.LayerNorm()(x)
        attn_out = x + Attention(self.config)(attn_out, training)

        mlp = [
            nn.Dense(4 * self.config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02)),
            nn.gelu,
            nn.Dense(self.config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.config.n_layer))),  # ref 3
            nn.Dropout(self.config.recid_pdrop, deterministic=not training),
        ]

        mlp_out = nn.LayerNorm()(attn_out)
        for fn in mlp:
            mlp_out = fn(mlp_out)
        output = attn_out + mlp_out

        return output


class Embedding(nn.Module):
    '''position and token embedding'''
    config: ModelConfig

    @nn.compact
    def __call__(self, idx, training=True):
        """idx: (b, t), the data is seq of number in this toy problem"""
        b, t = idx.shape

        position_embedding = nn.Embed(
            self.config.context_window, 
            self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02)
            )

        # (t,n_embd) > (1,t,n_embd)
        wpe = position_embedding(jnp.arange(t))[None,]

        token_embedding = nn.Embed(
            self.config.vocab_size, 
            self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02)
            )
        wte = token_embedding(idx)  # (b, t, n_embd)

        output = nn.Dropout(self.config.embd_pdrop)(
            wpe + wte, 
            deterministic=not training
        )

        return output


class GPT(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, training=True):
        x = Embedding(self.config)(x, training)

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x, training)

        x = nn.LayerNorm()(x)

        # output head
        logits = nn.Dense(self.config.vocab_size)(x)

        return logits


# reference
# param speical init
# 1. Dense default init https://flax.readthedocs.io/en/latest/_modules/flax/linen/linear.html
# 2. layernorm init default to ones and zeros https://flax.readthedocs.io/en/latest/_modules/flax/linen/normalization.html#LayerNorm
# 3. Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
#     #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
#     #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
#     #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
#     #
#     # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py