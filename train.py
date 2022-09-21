from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from utils import TrainConfig, TrainerCallbackEvent
from model import GPT


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: GPT,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        self.config = config

        # setup model, send to device, ref 1
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model = model.to(self.device)
        print(f"model is running on {self.device}")

        # setup dataloader
        self.setup_dataloader(train_dataset, test_dataset)

        # setup optimizer
        self.optimizer = self.config_optimizers()

        # callback library, dict[str, List[fn]], mapped event name to list of function
        # set so one can't add duplicated fn to an event
        self.callbacks = defaultdict(set)

    def setup_dataloader(self, train_dataset, test_dataset):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=torch.utils.data.RandomSampler(
                train_dataset, replacement=True, num_samples=int(1e10)
            ),  # ref 2
            shuffle=False,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=100, num_workers=0, drop_last=False
        )

    def config_optimizers(self):
        """
        Separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        decay, no_decay = set(), set()
        decay_module_type = (nn.Linear,)
        no_decay_module_type = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                full_param_name = f"{mn}.{pn}" if mn else pn  # ref 3

                if pn.endswith("bias"):
                    no_decay.add(full_param_name)
                elif pn.endswith("weight") and isinstance(m, decay_module_type):
                    decay.add(full_param_name)
                elif pn.endswith("weight") and isinstance(m, no_decay_module_type):
                    no_decay.add(full_param_name)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        # sanity check if all params are properly categorzied
        intersection = decay & no_decay
        assert len(intersection) == 0

        union = decay | no_decay
        assert len(param_dict.keys() - union) == 0

        # create per param otions, ref 4
        params = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],  # ref 5
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            params, lr=self.config.learning_rate, betas=self.config.betas
        )
        return optimizer

    def loss_fn(self, logits, y):
        """
        logits: (b, t, vocab_size)
        y: (b, t), index of vocab_size embedding, wte
        """
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1  # ref 6
        )  # (b * t, vocab_size), (b * t), which fits cross_entropy signature

        return loss

    def run(self):
        self.model.train()

        for self.n_iter, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # forward
            logits = self.model(x)
            self.loss = self.loss_fn(logits, y)

            # backprop
            self.optimizer.zero_grad()
            self.loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_norm_clip
            )
            self.optimizer.step()

            # house keeping
            self.trigger_callback(TrainerCallbackEvent.on_train_batch_end)

            if self.config.max_iters and self.n_iter >= self.config.max_iters:
                break

    @torch.no_grad()
    def eval(self, split: str, max_batch=50):
        assert split in ("train", "test")
        self.model.eval()

        loader = self.train_loader if split == "train" else self.test_loader
        n = loader.dataset.sequence_length
        show_mistakes_max = 5

        result = []

        # data example
        # x: tensor([0, 2, 0, 1, 1, 1, 0, 0, 1, 1, 1])
        # y: tensor([-1, -1, -1, -1, -1,  0,  0,  1,  1,  1,  2])
        for b, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            idx = x[:, :n]
            sol = y[:, -n:]

            # greedy argmax
            generated_seq = self.model.generate(idx, n, do_sample=False)
            pred = generated_seq[:, n:]  # take those generated sequence

            correct = (sol == pred).all(1).cpu()  # (b, )
            result.extend(correct)

            if b == max_batch - 1:
                break

        rt = torch.tensor(result, dtype=torch.float)
        print(
            f"{split} final score: {rt.sum()}/{len(result)} = {100*rt.mean():.2f}% correct"
        )

    def add_callback(self, on_event: TrainerCallbackEvent, fn):
        """
        fn(Trainer): the fn takes in Trainer object
            rely on purely side effects like print or manipulating tariner state
        """
        self.callbacks[on_event].add(fn)

    def trigger_callback(self, on_event: TrainerCallbackEvent):
        for fn in self.callbacks.get(on_event, set()):
            fn(self)


# reference
# 1. Many ways to handle model, data, activation, gradient and optimizer states.
#       Here I'll just put the whole model in CUDA and handle batch data in training runtime.
#       Check Zero for scaling. https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
# 2. sampler: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
# 3. because named_modules and named_parameters are recursive
#       we will see the same tensors p many many times. but doing it this way
#       allows us to know which parent module any tensor p belongs to...
#       - By default, if you do model.named_parameters(), the naming convention is module_name.param_name
# 4. per param option for optimizer: https://pytorch.org/docs/stable/optim.html#per-parameter-options
# 5. Parameters need to be specified as collections that have a deterministic
#       ordering that is consistent between runs. Examples of objects that don’t
#       satisfy those properties are sets and iterators over values of dictionaries.
#       - So order doesn't mater as long as you have the same order between runs.
# 6. ignore_index (int, optional), this is how we ignore input location my masking with -1, check dataset
#       Specifies a target value that is ignored and does not contribute to the input gradient.
#       https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
