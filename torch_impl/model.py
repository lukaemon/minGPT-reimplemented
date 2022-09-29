import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_impl.utils import ModelConfig


class CausalMultiheadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert (
            config.n_embd % config.n_head == 0
        ), "embedding space has to be multiple of n head"

        # input projection, batched
        self.kvq_proj = nn.Linear(config.n_embd, 3 * config.n_embd)

        # ref 4
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.c_drop = nn.Dropout(config.recid_pdrop)

        # full size mask
        self.register_buffer(
            "full_mask",
            torch.tril(torch.ones(config.context_window, config.context_window))[
                None, None, :, :
            ],
        )  # (1, 1, window, window)

    def forward(self, x):
        """
        x shape is the same as output (b, t, d_embd)

        torch.transpose: the given dimensions dim0 and dim1 are swapped,the order makes no difference.
        """
        b, t, c = x.size()  # (b, t, c=emb)
        assert (
            c == self.config.n_embd
        ), f"input embeding space({c}) ~= n_embd({self.config.n_embd}), did you forget the embedding layer?"

        # (b, t, c==emb)
        # -(proj)> (b, t, 3* emb)
        # -(tensor_split)> (b, t, h, hs=emb/h)
        # -(transpose)> (b, h, t, hs)
        q, k, v = self.kvq_proj(x).tensor_split(3, dim=-1)
        q = q.view(b, t, self.config.n_head, c // self.config.n_head).transpose(1, 2)
        k = k.view(b, t, self.config.n_head, c // self.config.n_head).transpose(1, 2)
        v = v.view(b, t, self.config.n_head, c // self.config.n_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / math.sqrt(c)  # (b, h, t, t)
        att.masked_fill_(self.full_mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (b, h, t, t) @ (b, h, t, hs) = (b, h, t, hs)

        # (b, h, t, hs) > (b, t, h, hs) > (b, t, c), multi-head concat
        out = out.transpose(1, 2).contiguous().view(b, t, c)

        out = self.c_drop(self.c_proj(out))
        return out


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mh_attn = CausalMultiheadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)

        # You need that naming convention. Nameless nn.Sequential or ModuleList won't do it, ref 4
        self.ff = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                act=nn.GELU("tanh"),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                c_drop=nn.Dropout(config.recid_pdrop),
            )
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

        # this is for ref 4. You can do the chaining in forward but the ugliness remain the same.
        self.ff_fn = lambda x: self.ff.c_drop(
            self.ff.c_proj(self.ff.act(self.ff.c_fc(x)))
        )

    def forward(self, x):
        x = x + self.mh_attn(self.ln1(x))
        x = x + self.ff_fn(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.context_window, config.n_embd),
                embd_drop=nn.Dropout(config.embd_pdrop),
                blks=nn.ModuleList(
                    Block(config) for _ in range(config.n_layer)
                ),  # ref 6
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.register_buffer("position", torch.arange(config.context_window))  # ref 5

        # output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init params
        self.apply(self._init_weights)  # ref 3

        # special init, ref 4
        for name, p in self.transformer.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # counting number of params
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.3fM" % (n_params / 1e6))

    def _init_weights(self, module):  # ref 3
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        """
        idx (b, t): name comes from embedding convention.
            > nn.Embedding is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.

        output logits: (b, t, vocab_size)
        """
        b, t = idx.size()
        assert (
            t <= self.config.context_window
        ), f"{self.config.context_window=}, input idx {t=}"

        # embedding
        pos_emb = self.transformer.wpe(self.position[:t])[None]  # (1, t, n_embd)
        token_emb = self.transformer.wte(idx)  # (b, t, n_embd)

        # forward
        x = self.transformer.embd_drop(pos_emb + token_emb)
        for blk in self.transformer.blks:
            x = blk(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, do_sample=False, top_k=None):
        """
        idx: (b, t), conditioned sequence input as idx to embeddings
        max_new_tokens: how many new tokens to generate
        """
        self.eval()

        for _ in range(max_new_tokens):
            # take the last context_window tokens
            idx_capped = idx[:, -self.config.context_window :]

            logits = self(idx_capped)  # (b, t, vocab_size)

            # (b, vocab_size), take the logits of last token as prediction
            pred = logits[:, -1, :]  # (b, vocab_size)

            if top_k:
                # first to last column, largest to smallest
                v, _ = torch.topk(pred, top_k)  # (b, top_k)

                # v[:, [-1]], (b, 1), take last column with smallest value
                # fill -inf to the postion with smaller value to the the smallest top_k
                pred.masked_fill_(pred < v[:, [-1]], float("-inf"))

            # -inf previously would be 0 prob here
            prob = F.softmax(pred, dim=-1)  # (b, vocab_size)

            if do_sample:  # sample with prob
                idx_next = torch.multinomial(prob, num_samples=1)  # (b, 1), ref 7
            else:  # take the top prediction
                _, idx_next = torch.topk(prob, k=1)  # (b, 1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# reference
# 1. per GPT2 paper: an additional layer normalization was added after the final selfattention block
# 2. bias=False
#     - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L951
# 3. Applies fn recursively to every submodule (as returned by .children()) as well as self. Typical use includes initializing the parameters of a model
#     - https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply
#     - I can appreciate JAX's handling of pytree, such as jax.tree_map
#     - https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
# 4. Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
#     #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
#     #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
#     #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
#     #
#     # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
# 5. use register buffer to avoid ugly device hack in forward at runtime
#     # device = idx.device
#     # pos = torch.arange(
#     #     t, device=device, dtype=torch.long
#     # )  # (t,) generated in runtime, ugly way to comply with input's device
#     # pos_emb = self.transformer.wpe(pos)[None]  # (1, t, n_embd)
# 6. TypeError: GPT.__init__.<locals>.<genexpr> is not a Module subclass
#     - nn.Sequential won't work
# 7. If input is a matrix with m rows, out (m, num_samples)
#     https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial#torch.multinomial
