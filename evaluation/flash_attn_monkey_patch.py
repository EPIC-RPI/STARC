from typing import List, Optional, Tuple

import torch
from torch import nn
import math

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        padding_mask=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Attention forward with optional FlashAttention acceleration.

    Uses standard attention for token-by-token decoding (or when Q/K lengths differ), and
    uses FlashAttention for full-sequence attention when applicable.
    """
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if query_states.shape[-2] == 1 or query_states.shape[-2] != key_states.shape[-2]:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)

    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                indices, bsz, q_len
            ),
            'b s (h d) -> b s h d', h=nheads
        )

    attn_output = self.o_proj(rearrange(output, 'b s h d -> b s (h d)'))
    return attn_output, None, past_key_value


def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None
):
    """
    Keep the decoder attention mask aligned with the key padding mask for FlashAttention.

    For prefill (seq_len > 1, no past), return the input mask directly; otherwise, fall back
    to the default Bart decoder mask preparation.
    """
    if input_shape[-1] > 1 and past_key_values_length == 0:
        return attention_mask
    return transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length
    )


def replace_llama_attn_with_flash_attn():
    """Install FlashAttention-backed forward and matching mask preparation for Llama."""
    print('use FlashAttention')
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward


def replace_mistral_attn_with_flash_attn():
    """Install FlashAttention-backed forward and matching mask preparation for Mistral."""
    print('use FlashAttention')
    transformers.models.mistral.modeling_mistral.MistralModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward
