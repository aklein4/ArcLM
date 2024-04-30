from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from transformers import LongT5Config
from transformers.models.longt5.modeling_longt5 import LongT5PreTrainedModel, LongT5Stack, LongT5Attention

from utils.data_utils import DotDict


__ARC_ATTENTION_SWITCH__ = False

class arc_attention_switch:
    def __init__(self, active):
        self.active = active
    def __enter__(self):
        global __ARC_ATTENTION_SWITCH__
        __ARC_ATTENTION_SWITCH__ = self.active
    def __exit__(self):
        global __ARC_ATTENTION_SWITCH__
        __ARC_ATTENTION_SWITCH__ = False

def arc_compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device
    if __ARC_ATTENTION_SWITCH__:
        print("switch")
        context_position = torch.arange(query_length//2, dtype=torch.long, device=device)
        memory_position = torch.arange(key_length//2, dtype=torch.long, device=device)
        
        context_position = torch.cat([context_position]*2, dim=-1)[:, None]
        memory_position = torch.cat([memory_position]*2, dim=-1)[None, :]
    else:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]  
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values

LongT5Attention.compute_bias = arc_compute_bias


class ArcLongT5(LongT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
        r"arc_head.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: LongT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        # init shared embedding
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # init encoder
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LongT5Stack(encoder_config, self.shared)

        # init decoder
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = LongT5Stack(decoder_config, self.shared)

        # init heads
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.arc_head = nn.Linear(config.d_model, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        encoder_outputs: Tuple[Tuple[torch.Tensor]],
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        arc_is_active: bool = False,
    ) -> DotDict:

        hidden_states = encoder_outputs[0]

        # Decode
        with arc_attention_switch(arc_is_active):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
            )
        sequence_output = decoder_outputs[0]
        
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        lm_input = sequence_output
        if self.config.tie_word_embeddings:
            lm_input = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(lm_input)

        # Compute arc output
        arc_output = self.arc_head(sequence_output)

        return DotDict(
            lm_logits=lm_logits,
            arc_output=arc_output,
            encoder_outputs=encoder_outputs,
        )


    """
    Utilities.
    """
    
    def init_arc_head(self):
        self.arc_head.reset_parameters()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    """
    The below functions are of unknown purpose.
    """


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past