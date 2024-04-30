import torch
import torch.nn.functional as F

import numpy as np

from utils.data_utils import DotDict


def lm_metrics(input_ids, logits, padding_mask=None):

    # make sure logits are normalized
    logits = F.log_softmax(logits, dim=-1)

    # offset the input_ids by one
    logits = logits[:, :-1]
    input_ids = input_ids[:, 1:]
    padding_mask = torch.logical_not(padding_mask[:, 1:])

    # calculate the logp
    logp = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids.reshape(-1),
        reduction='none'
    ).reshape(logits.shape[0], logits.shape[1])

    # apply the padding mask
    masked_logits = logits.reshape(-1, logits.shape[-1])[padding_mask.reshape(-1)]
    masked_input_ids = input_ids.reshape(-1)[padding_mask.reshape(-1)]
    masked_logp = logp.reshape(-1)[padding_mask.reshape(-1)]

    # loss is negative mean logp
    loss = -masked_logp.sum() / padding_mask.numel()

    # bpb is the negative mean logp / log(2)
    bpb = -masked_logp.mean() / np.log(2)

    # perplexity is exp(mean logp) for each sequence
    ppl = 0
    for i in range(logits.shape[0]):
        curr_logp = -logp[i][padding_mask[i]]
        ppl = ppl + torch.exp(curr_logp.mean())
    ppl = ppl / logits.shape[0]

    # accuracy is the percentage of correct tokens
    acc = (
        torch.argmax(masked_logits, dim=-1) ==
        masked_input_ids
    ).float().mean()

    return DotDict(
        loss=loss,
        bpb=bpb,
        ppl=ppl,
        acc=acc
    )