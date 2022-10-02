# Prologue
Minimal reimplementation of [Andrej Karparthy's minGPT](https://github.com/karpathy/minGPT), which is a very good end to end practice for decoder only transformer. 

I read, understand then rewrite most of them, copy some, ignore some, rearange, and makes few tweaks here and there, which are reflection to where I am, wrt to my understanding of transformer, ML workflow, and coding overall. 

Thanks Andrej. I feel like reading a good book. The intentional simplicity and extra comments really help.


# Mental framework
| full stack AI         | in abstract                 |
| ----------------------------- | --------------------------- |
| model                     | form of intelligence      |
| dataset                   | online, offline experiences, dynamic simulation |
| loss function, reward | feedback loop engineering              |
| optimizer                 | making change according to feedbacks               |
| training                  | the whole process           |
| inference                 | should merge with training to form a continuous process                            |

Coding up model from a paper is an interesting, yet tiny step of the full stack. The invention of useful model is genuine effort. The simplicity of transformer is beautiful. 


# General notes
- This is a toy implementation for learning purpose, just ignore the integration with huggingface. However, the `from_pretrained` method is a good example for taking existing weights and adapt to your own model. 
- Ignore `bpe encoding`. The model can only handle toy, synthetic dataset, or char level dataset. I believe tokenization strikes a practical balance between utf-8 and word, but soon would be prematured optimization. Here is the quote from [PerceiverAR](https://arxiv.org/abs/2202.07765) on tokenization
> Tokenization is a broadly useful strategy, but it has its downsides. For many domains, effective tokenization schemes rely on lossy compression. Data is discarded in the tokenization process, and inputs can not be recovered exactly. Care is required to ensure that the data can be reconstructed at a fidelity adequate for the required application. Neural compression schemes such as VQ-VAE require users to train and maintain additional encoder and decoder networks, which can hinder ready application to new datasets and domains. And, perhaps most tellingly, effective tokenization is typically designed and used in a domain-specific fashion, which limits the ease of adaption and scaling to new domains.


# PyTorch implementation related
- Special parameter init really force me to bake some naming convention in model building. (model.py ref 4) What's the better way of handing this? 
- The original implementation bundles `loss` computation and `optimizer` configuration into model. 
    - `loss fn` should be a separate entities?
    - `optimizer` configuration should be separated as well. This optimizer has different weight decay treatment toward different kind of parameters, though the differentiation is done by module type and parameter name, which are not tied to the model specifics. 
- Training process has to put everything together. Make sense to build a trainer class to bundle all relevant states and functions together. Still can't properly evaluate the pros and cons of using framework like `Pytorch Lightening`. Maybe I should bypass it if I'm heading to `DeepSpeed` and `Ray` soon after, assuming default Lightening's integration with DeepSpeed is not what I need and have to go a level deeper of abstraction. Anyway, I'll know when I hit the wall of current tools. 
- The callback system is intuitive and interesting.


# JAX implementation related
- `Flax` model code is simpler because `@nn.compact` saves the `init` method. 
- Model definition is more like building nested functions with JAX. The goal is to write it in a **JIT-able** way to exploit `XLA`'s power. This is not as intuitive as `PyTorch`, but the hope is small investment to comply with the functional style here would yield out-proportioned engineering benefits later. The following is the non-intuitive part compared to `PyTorch` but one could get used to easily.
    - Immutable tensor
    - Functional paradigm forces explicit separation between model and parameters. 
    - Explicit care of random number generator. 
- I assume the clear separation between functions and parameters would make decentralized training and inference easier. Data, model, pipeline, tensor parallelism and [various stage of ZeRO](https://www.microsoft.com/en-us/research/blog/ZeRO-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/). Would have to dig deep into `deepspeed` and `pjit` to know better. 


# Cooking log
## 20221002
JAX version is working. However, my inference speed after JIT is still slow.  
I don't know if that's normal for autoregressive inference, or I didn't JIT it right. Assuming my ignorance is playing me. 

Experiment with various dataset complexity. 
- `gpt-nano` setup 32 epoch works until vocab_size and seq_len = 32
- `gpt-mini` setup, 96 epoch works for vocab, seq_len = 64. Train longer... Stopped here, training time is growing exponentially.

## 20220921
First error free training and evaluation. But my loss is way higher than Andrej's. The party begins. Have to find out hidden bugs that didn't crush the runtime but eating away performance in the dark. The beauty of ML alchemy.

Andrej:
``` 
iter_dt 9.03ms; iter 1500: train loss 0.00048
iter_dt 8.87ms; iter 1600: train loss 0.00044
iter_dt 9.01ms; iter 1700: train loss 0.01302
iter_dt 8.93ms; iter 1800: train loss 0.00352
iter_dt 9.01ms; iter 1900: train loss 0.01247
```

mine: 
```
trainer.n_iter=1500, trainer.loss.item()=0.1127
trainer.n_iter=1600, trainer.loss.item()=0.1014
trainer.n_iter=1700, trainer.loss.item()=0.1729
trainer.n_iter=1800, trainer.loss.item()=0.1307
trainer.n_iter=1900, trainer.loss.item()=0.1604
trainer.n_iter=2000, trainer.loss.item()=0.1180
```

- Just found out all of my linear module's `bias=False`. Total params are lesser. Why? It should default to be true. 
  - I did `module.bias = None` for special init. If I change it to `torch.nn.init.zeros_(module.bias)`, biases are back..... but the loss is worse. The problem is not the lack of bias. Good find though. 
- I did post `layernorm`. Should be pre `layernorm`. **This is it!!** Pre layernorm performs significantly better. I can relate to openai's smile when they changed to pre-layernorm from gpt to gpt2. 
```python
# wrong
def forward(self, x):
    x = self.mh_attn(x)
    x = x + self.ln1(x)
    x = self.ff_fn(x)
    x = x + self.ln2(x)

# correct
def forward(self, x):
    x = x + self.mh_attn(self.ln1(x))
    x = x + self.ff_fn(self.ln2(x))
```