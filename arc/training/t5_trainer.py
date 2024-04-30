import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

from transformers.optimization import Adafactor

from training.base_trainer import BaseTrainer

from utils.training_utils import lm_metrics
from utils.data_utils import DotDict
import utils.constants as constants
from utils.arc_utils import get_arc_input_ids, get_arc_attention_mask, get_arc_metrics


class T5Trainer(BaseTrainer):

    _log_file = os.path.join(constants.LOCAL_DATA_PATH, "log.csv")
    _progress_file = os.path.join(constants.LOCAL_DATA_PATH, "progress.png")

    _hyperparams = [
        "lr",
        "bs",
        "num_steps",
        "warmup_steps",
        "eval_freq",
        "checkpoint_freq",
        "dtype",
        "max_length",
        "max_eval_examples",
    ]

    _metrics = ["loss", "bpb", "ppl", "acc", "arc_loss", "arc_acc"]

    def __init__(
        self,
        save_name,
        **kwargs
    ):
        super().__init__(save_name, **kwargs)

        self.log = DotDict()
        self.reset_log()


    def reset_log(self):
        self.log = DotDict(
            train=DotDict(),
            eval=DotDict()
        )

        for m in self._metrics:
            self.log.train[m] = []
            self.log.eval[m] = []


    def upload(self):

        # save log
        df = pd.DataFrame(self.log.eval.to_dict())
        df.to_csv(self._log_file)

        # get rolling training metrics
        df = pd.DataFrame(self.log.train.to_dict())
        roll = df.rolling(window=self.eval_freq, center=False, min_periods=self.eval_freq//2)
        mask = ~np.isnan(roll["loss"].mean())
        x = np.arange(len(roll["loss"].mean()))[mask]

        # plot simple metrics
        fig, ax = plt.subplots(2, len(self._metrics), figsize=(5*len(self._metrics), 10))

        # plot training metrics
        for i, metric in enumerate(self._metrics):
            ax[0,i].plot(x, roll[self._metrics[i]].mean()[mask])
            ax[0,i].set_title(f"Training {metric.upper()}")

        # plot eval metrics
        for i, metric in enumerate(self._metrics):
            ax[1,i].plot(self.log.eval[metric])
            ax[1,i].set_title(f"Eval {metric.upper()}")

        # finish plot
        plt.suptitle(f"Arc Training Progress ({len(self.log.train.loss)} steps)")
        plt.tight_layout()
        plt.savefig(self._progress_file)
        plt.close()

        super().upload(self._log_file, self._progress_file)


    def _get_tokens(self, loader, tokenizer):
        a, b = loader(self.bs)

        x_a = tokenizer(
            a,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(constants.DEVICE)
        x_b = tokenizer(
            b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(constants.DEVICE)

        return DotDict(
            input_ids=x_a.input_ids,
            input_mask=x_a.attention_mask,
            decoder_input_ids=x_b.input_ids,
            decoder_attention_mask=x_b.attention_mask,
        )


    @torch.no_grad()
    def evaluate(
        self,
        tokenizer,
        model,
        val_loader
    ):
        model.eval()

        tmp_log = DotDict()
        for m in self._metrics:
            tmp_log[m] = []

        examples = 0
        val_loader.reset()
        with tqdm(desc="Evaluating", leave=False) as pbar:
            while not val_loader.done:

                enable_autocast = self.dtype != torch.float32
                with torch.autocast(
                    device_type=str(constants.DEVICE),
                    dtype=(torch.float16 if not enable_autocast else self.dtype),
                    enabled=enable_autocast
                ):

                    # handle inputs
                    x = self._get_tokens(val_loader, tokenizer)

                    # get reusable encodings
                    encoder_outputs = model.encode(
                        x.input_ids,
                        x.attention_mask
                    )

                    # get pos/neg samples
                    logits = model(
                        x.decoder_input_ids,
                        encoder_outputs,
                        x.decoder_attention_mask,
                        x.attention_mask,
                    ).lm_logits

                    arc_ids = get_arc_input_ids(
                        x.decoder_input_ids,
                        logits
                    )
                    arc_mask = get_arc_attention_mask(
                        arc_ids.shape,
                        arc_ids.device
                    )
                    
                    # get predictions
                    model_out = model(
                        arc_ids,
                        encoder_outputs,
                        arc_mask,
                        x.attention_mask,            
                    )
                    arc_metrics = arc_metrics(
                        model_out.arc_output,
                        x.decoder_attention_mask<0.5
                    )
                    metrics = lm_metrics(
                        x.input_ids, model_out.lm_logits[:, :x.input_ids.shape[-1]],
                        x.decoder_attention_mask<0.5
                    )

                    # save metrics
                    for m in self._metrics:
                        try:
                            tmp_log[m].append(metrics[m].item())
                        except:
                            tmp_log[m].append(arc_metrics[m].item())

                pbar.set_postfix({str(k): np.mean(v) for k, v in tmp_log.items()})
                pbar.update(self.bs)

                examples += self.bs
                if examples >= self.max_eval_examples:
                    break

        # save metrics
        for m in self._metrics:
            self.log.eval[m].append(np.mean(tmp_log[m]))


    def train(
        self,
        tokenizer,
        model,
        train_loader,
        val_loader
    ):

        for p in model.parameters():
            p.requires_grad = True

        optimizer = Adafactor(model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        scaler = torch.cuda.amp.GradScaler()

        self.evaluate(tokenizer, model, val_loader)

        with tqdm(range(self.num_steps), desc="Training") as pbar:
            for step in pbar:

                model.train()

                enable_autocast = self.dtype != torch.float32
                with torch.autocast(
                    device_type=str(constants.DEVICE),
                    dtype=(torch.float16 if not enable_autocast else self.dtype),
                    enabled=enable_autocast
                ):

                    # handle inputs
                    x = self._get_tokens(train_loader, tokenizer)

                    # get reusable encodings
                    encoder_outputs = model.encode(
                        x.input_ids,
                        x.attention_mask
                    )

                    # get pos/neg samples
                    with torch.no_grad():
                        logits = model(
                            x.decoder_input_ids,
                            encoder_outputs,
                            x.decoder_attention_mask,
                            x.attention_mask,
                        ).lm_logits

                        arc_ids = get_arc_input_ids(
                            x.decoder_input_ids,
                            logits
                        )
                        arc_mask = get_arc_attention_mask(
                            arc_ids.shape,
                            arc_ids.device
                        )
                    
                    # get predictions
                    model_out = model(
                        arc_ids,
                        encoder_outputs,
                        arc_mask,
                        x.attention_mask,            
                    )
                    arc_metrics = arc_metrics(
                        model_out.arc_output,
                        x.decoder_attention_mask<0.5
                    )
                    metrics = lm_metrics(
                        x.input_ids, model_out.lm_logits[:, :x.input_ids.shape[-1]],
                        x.decoder_attention_mask<0.5
                    )

                    loss = arc_metrics.loss + metrics.loss
                
                if enable_autocast:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(True)
                lr_scheduler.step()

                # save metrics
                for m in self._metrics:
                    try:
                        self.log.train[m].append(metrics[m].item())
                    except:
                        self.log.train[m].append(arc_metrics[m].item())
                pbar.set_postfix({k: v.item() for k, v in metrics.items()})

                if (step+1) % self.eval_freq == 0 or step == self.num_steps-1:
                    self.evaluate(tokenizer, model, val_loader)
                    self.upload()

                if (step+1) % self.checkpoint_freq == 0 or step == self.num_steps-1:
                    self.save_checkpoint(
                        {
                            "model": model
                        }
                    )
                
