import torch
import torch.nn.functional as F

from utils.data_utils import DotDict


def get_arc_input_ids(
    input_ids,
    logits
):
    fake_input_ids = input_ids.clone()

    # sample fake input_ids
    dist = torch.distributions.Categorical(logits=logits)
    fakes = dist.sample()

    # replace the second half of the input_ids (we don't care about pad tokens)
    fake_input_ids[:, 1:] = fakes[:, :-1]

    return torch.cat(
        [
            input_ids,
            fake_input_ids
        ],
        dim=-1
    )


def get_arc_attention_mask(
    input_shape,
    device
):
    """
    Create attention mask for ARC task.
    """
    bs, seq_len = tuple(input_shape)
    assert seq_len % 2 == 0, "Sequence length must be even."
    half_len = seq_len // 2

    base_mask = torch.ones(
        bs,
        half_len,
        half_len,
        device=device,
    )

    nw_mask = torch.triu(base_mask, diagonal=1)
    ne_mask = base_mask
    sw_mask = torch.triu(base_mask, diagonal=0)
    se_mask = 1-torch.diag_embed(torch.diagonal(base_mask, 0, -2, -1))

    return 1-torch.cat(
        [
            torch.cat([nw_mask, ne_mask], dim=-1),
            torch.cat([sw_mask, se_mask], dim=-1)
        ],
        dim=-2
    )


def get_arc_metrics(
    arc_outputs,
    padding_mask
):
    assert arc_outputs.shape[1] % 2 == 0, "ARC outputs must be even."

    real_outputs, fake_outputs = torch.split(
        arc_outputs,
        arc_outputs.shape[1]//2,
        dim=1
    )
    assert tuple(padding_mask.shape) == tuple(real_outputs.shape[:-1]), "Padding mask must match ARC outputs shape."

    # remove first token
    real_outputs = real_outputs[:, 1:]
    fake_outputs = fake_outputs[:, 1:]
    padding_mask = padding_mask[:, 1:]

    # apply padding
    padding_mask = ~padding_mask.reshape(-1)
    real_outputs = real_outputs.reshape(-1)[padding_mask]
    fake_outputs = fake_outputs.reshape(-1)[padding_mask]

    # loss is CNE loss
    loss = -(
        F.logsigmoid(real_outputs) +
        F.logsigmoid(-fake_outputs)
    ).sum()
    loss = loss / (2*padding_mask.numel())

    # accuracy is the number of correct predictions
    acc = (
        (real_outputs > 0).float() +
        (fake_outputs < 0).float()
    ).mean()

    return DotDict(
        arc_loss=loss,
        arc_acc=acc
    )