import math
import numpy as np
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.utils.dlpack import to_dlpack

import types

from transformers.cache_utils import DynamicCache

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.models.mistral.modeling_mistral import (
    MistralAttention, MistralFlashAttention2, MistralSdpaAttention,
    apply_rotary_pos_emb as mistral_apply_rotary_pos_emb,
    repeat_kv as mistral_repeat_kv,
)


def cluster_heavy_hitter_mask(
    query_states: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    reorder_map: torch.Tensor,
    B: int,
    L_orig: int,
    num_groups: int
) -> torch.Tensor:
    """
    Compute a per-query-head token-selection mask based on centroid scoring and cluster-aware budgeting.

    The implementation supports grouped-query attention (GQA) by broadcasting KV-head clustering metadata
    to the corresponding Q-head groups, preserving recent tokens (label == -1), then selecting full
    clusters by score and truncating the boundary cluster to fit the remaining budget.

    Returns:
        A boolean mask of shape [batch, num_q_heads, q_len, L_orig] in the original token order.
    """
    device = query_states.device
    Bsz, H_q, q_len, D = query_states.shape
    _, H_k, C, _ = centroids.shape
    assert H_q == H_k * num_groups, "Grouped-attention assumption violated"

    centroids_q = centroids.repeat_interleave(num_groups, dim=1)
    labels_q = labels.repeat_interleave(num_groups, dim=1)
    reorder_q = reorder_map.repeat_interleave(num_groups, dim=1)

    labels_pos_q = labels_q.clamp(min=0)
    valid_mask = (labels_q >= 0).int()

    scores = torch.einsum(
        "bhqd,bhcd->bhqc",
        query_states.float(),
        centroids_q.float()
    ) / math.sqrt(D)
    cluster_scores = scores.max(dim=2).values
    sorted_idx = torch.argsort(cluster_scores, dim=-1, descending=True)

    pad_cnt_k = (labels == -1).int().sum(-1)
    pad_cnt_q = pad_cnt_k.repeat_interleave(num_groups, dim=1)
    budget_np = B - pad_cnt_q

    cluster_cnts_k = torch.zeros(Bsz, H_k, C, dtype=torch.int32, device=device)
    cluster_cnts_k.scatter_add_(-1, labels.clamp(min=0), valid_mask)
    cluster_cnts_q = cluster_cnts_k.repeat_interleave(num_groups, dim=1)

    ordered_cnts = torch.gather(cluster_cnts_q, -1, sorted_idx)
    cumsum_cnts = ordered_cnts.cumsum(-1)
    keep_cluster_full = cumsum_cnts <= budget_np.unsqueeze(-1)

    chosen_full = torch.zeros_like(keep_cluster_full, dtype=torch.bool)
    chosen_full.scatter_(-1, sorted_idx, keep_cluster_full)
    keep_token_full = (labels_q == -1) | chosen_full.gather(-1, labels_pos_q)

    chosen_cnt_clusters = (keep_token_full & (labels_q != -1)).int().sum(-1)
    leftover = (budget_np - chosen_cnt_clusters).clamp(min=0)

    boundary_mask = (cumsum_cnts > budget_np.unsqueeze(-1)) & \
                    ((cumsum_cnts - ordered_cnts) < budget_np.unsqueeze(-1))
    boundary_idx_sorted = boundary_mask.float().argmax(-1)
    boundary_cluster = sorted_idx.gather(
        -1, boundary_idx_sorted.unsqueeze(-1)).squeeze(-1)

    in_boundary = labels_pos_q == boundary_cluster.unsqueeze(-1)
    rank_in_cluster = torch.cumsum(in_boundary.int(), dim=-1)
    partial_keep = in_boundary & (rank_in_cluster <= leftover.unsqueeze(-1))

    keep_token = keep_token_full | partial_keep

    mask = torch.zeros(Bsz, H_q, q_len, L_orig, dtype=torch.bool, device=device)
    mask.scatter_(
        -1,
        reorder_q.unsqueeze(2).expand(-1, -1, q_len, -1),
        keep_token.unsqueeze(2).expand(-1, -1, q_len, -1)
    )
    return mask


@torch.no_grad()
def cluster_prefill_kv(
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    interval: int,
    chunk_size: int,
    n_iter: int = 16
):
    """
    Cluster prefill KV in sliding windows and concatenate per-window results.

    Each window is clustered independently and then merged by applying global offsets to the
    reorder map and labels to keep indices consistent across windows.

    Returns:
        (k_all, v_all, map_all, lab_all, cen_all) with the same format as cluster_on_every_head.
    """
    bsz, H, L_total, D = k_full.shape
    assert bsz == 1, "Currently only batch_size=1 is supported"

    k_out, v_out, map_out, label_out, centroids_out = [], [], [], [], []
    token_offset, label_offset = 0, 0

    for start in range(0, L_total, interval):
        end = min(start + interval, L_total)
        k_slice = k_full[:, :, start:end, :]
        v_slice = v_full[:, :, start:end, :]

        (
            k_c, v_c, map_c, label_c, centroids_c
        ) = cluster_on_every_head(
            k_slice, v_slice,
            chunk_size=chunk_size,
            n_iter=n_iter
        )

        map_c = map_c + token_offset
        label_c = label_c + label_offset

        token_offset += (end - start)
        label_offset += centroids_c.size(2)

        k_out.append(k_c)
        v_out.append(v_c)
        map_out.append(map_c)
        label_out.append(label_c)
        centroids_out.append(centroids_c)

    k_all = torch.cat(k_out, dim=2)
    v_all = torch.cat(v_out, dim=2)
    map_all = torch.cat(map_out, dim=2)
    lab_all = torch.cat(label_out, dim=2)
    cen_all = torch.cat(centroids_out, 2)

    return k_all, v_all, map_all, lab_all, cen_all


@torch.no_grad()
def batched_kmeans(keys: torch.Tensor, n_clusters: int, n_iter: int = 16):
    """
    Run Lloyd's k-means independently per head.

    Args:
        keys: [num_heads, seq_len, head_dim], expected to be L2-normalized.
        n_clusters: number of clusters per head.
        n_iter: maximum Lloyd iterations.

    Returns:
        labels: [num_heads, seq_len]
        centroids: [num_heads, n_clusters, head_dim]
    """
    H, L, D = keys.shape
    centroids = keys[:, :n_clusters, :].clone()

    for _ in range(n_iter):
        x_norm = (keys ** 2).sum(-1, keepdim=True)
        c_norm = (centroids ** 2).sum(-1).unsqueeze(-2)
        dist2 = x_norm + c_norm - 2.0 * torch.matmul(keys, centroids.transpose(-1, -2))

        labels = dist2.argmin(-1)

        centroids_sum = torch.zeros_like(centroids)
        centroids_count = torch.zeros(H, n_clusters, device=keys.device, dtype=torch.int32)

        centroids_sum.scatter_add_(
            1,
            labels.unsqueeze(-1).expand(-1, -1, D),
            keys
        )

        centroids_count.scatter_add_(
            1,
            labels,
            torch.ones_like(labels, dtype=torch.int32)
        )

        centroids_count_clamped = centroids_count.clamp_min(1).unsqueeze(-1)
        centroids_new = centroids_sum / centroids_count_clamped

        if torch.allclose(centroids, centroids_new, atol=1e-4):
            centroids = centroids_new
            break
        centroids = centroids_new

    return labels, centroids


@torch.no_grad()
def cluster_on_every_head(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    chunk_size: int,
    n_iter: int = 16,
):
    """
    Cluster KV per head and reorder tokens by cluster assignment.

    Returns:
        keys_sorted:  [1, H, L, D]
        values_sorted:[1, H, L, D]
        reorder_map:  [1, H, L]  (indices into original order)
        labels_new:   [1, H, L]  (cluster labels in sorted order)
        centroids:    [1, H, C, D]
    """
    device = key_states.device
    bsz, num_heads, seq_len, head_dim = key_states.shape
    assert bsz == 1, "Only batch_size=1 is supported"

    n_clusters = (seq_len + chunk_size - 1) // chunk_size

    keys_norm = F.normalize(key_states[0], p=2, dim=-1)
    labels, centroids = batched_kmeans(keys_norm, n_clusters=n_clusters, n_iter=n_iter)

    sorted_idx = labels.argsort(dim=1)
    labels_sorted = labels.gather(1, sorted_idx)
    gather_idx = sorted_idx.unsqueeze(-1).expand(-1, -1, head_dim)

    keys_sorted = key_states[0].gather(1, gather_idx)
    values_sorted = value_states[0].gather(1, gather_idx)

    reorder_map = sorted_idx
    labels_new = labels_sorted
    centroids_t = centroids

    return (
        keys_sorted.unsqueeze(0),
        values_sorted.unsqueeze(0),
        reorder_map.unsqueeze(0).to(torch.long),
        labels_new.unsqueeze(0).to(torch.long),
        centroids_t.unsqueeze(0),
    )


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    output_attentions: bool = False,
    use_cache: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:

    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    if past_key_value is not None:
        if len(past_key_value) == 2:
            old_k, old_v = past_key_value
            original_k, original_v = past_key_value
            old_clustered_k = None
            old_clustered_v = None
            old_reorder_map = None
            decode_step = 0
            new_token_count = 0
            old_labels = None
            cluster_centers = None
        elif len(past_key_value) == 11:
            original_k, original_v, old_k, old_v, old_clustered_k, old_clustered_v, old_reorder_map, decode_step, new_token_count, old_labels, cluster_centers = past_key_value
        else:
            raise ValueError("Unsupported past_key_value format!")
    else:
        old_k, old_v = None, None
        original_k, original_v = None, None
        old_clustered_k, old_clustered_v = None, None
        old_reorder_map = None
        decode_step, new_token_count = 0, 0
        old_labels = None
        cluster_centers = None

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    decode_step += 1
    new_token_count += 1

    kv_seq_len = key_states.shape[-2]
    if original_k is not None:
        kv_seq_len += original_k.shape[-2]

    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if original_k is not None:
        key_states_temp = torch.cat([original_k, key_states], dim=2)
        value_states_temp = torch.cat([original_v, value_states], dim=2)
    else:
        key_states_temp = key_states
        value_states_temp = value_states

    if old_k is not None:
        key_states = torch.cat([old_k, key_states], dim=2)
        value_states = torch.cat([old_v, value_states], dim=2)

    new_past_k = key_states
    new_past_v = value_states

    if decode_step == 1:
        old_clustered_k, old_clustered_v, old_reorder_map, old_labels, cluster_centers = cluster_prefill_kv(
            key_states,
            value_states,
            interval=self.prefill_cluster_interval,
            chunk_size=self.prefill_chunk_size,
            n_iter=16
        )
        new_past_k = None
        new_past_v = None
        key_states = key_states[:, :, :0, :]
        value_states = value_states[:, :, :0, :]
        new_token_count = 0
    else:
        if new_token_count >= 64:
            new_clustered_k, new_clustered_v, new_reorder_map, new_labels, new_cluster_centers = cluster_on_every_head(
                key_states, value_states, self.chunk_size
            )

            if old_reorder_map is not None and old_reorder_map.numel() > 0:
                offset_map = old_reorder_map.max().item() + 1
            else:
                offset_map = 0
            new_reorder_map = new_reorder_map + offset_map

            if old_labels is not None and old_labels.numel() > 0:
                offset_label = old_labels.max().item() + 1
            else:
                offset_label = 0
            new_labels = new_labels + offset_label

            if old_clustered_k is not None and old_clustered_v is not None:
                updated_k = torch.cat([old_clustered_k, new_clustered_k], dim=2)
                updated_v = torch.cat([old_clustered_v, new_clustered_v], dim=2)
                updated_map = torch.cat([old_reorder_map, new_reorder_map], dim=2)
                updated_label = torch.cat([old_labels, new_labels], dim=2)
            else:
                updated_k = new_clustered_k
                updated_v = new_clustered_v
                updated_map = new_reorder_map
                updated_label = new_labels

            old_clustered_k = updated_k
            old_clustered_v = updated_v
            old_reorder_map = updated_map
            new_past_k = None
            new_past_v = None
            key_states = key_states[:, :, :0, :]
            value_states = value_states[:, :, :0, :]
            old_labels = updated_label
            cluster_centers = torch.cat([cluster_centers, new_cluster_centers], dim=2)
            new_token_count = 0

    if old_clustered_k is not None and old_clustered_v is not None:
        L_new = key_states.shape[2]
        if old_reorder_map.numel() > 0:
            current_offset = old_reorder_map.max().item() + 1
        else:
            current_offset = 0

        new_reorder_map_identity = torch.arange(
            current_offset,
            current_offset + L_new,
            device=old_reorder_map.device
        ).unsqueeze(0).unsqueeze(0)
        bsz = key_states.shape[0]
        num_heads = key_states.shape[1]
        new_reorder_map_identity = new_reorder_map_identity.expand(bsz, num_heads, L_new)
        final_map = torch.cat([old_reorder_map, new_reorder_map_identity], dim=2)
        key_states = torch.cat([old_clustered_k, key_states], dim=2)
        value_states = torch.cat([old_clustered_v, value_states], dim=2)

        if old_labels is not None:
            new_label_segment = torch.full(
                (bsz, num_heads, L_new),
                fill_value=-1,
                dtype=old_labels.dtype,
                device=old_labels.device,
            )
            final_labels = torch.cat([old_labels, new_label_segment], dim=2)

    if old_clustered_k is not None and old_clustered_v is not None:
        mask_bottom_original = cluster_heavy_hitter_mask(
            query_states=query_states,
            centroids=cluster_centers,
            labels=final_labels,
            reorder_map=final_map,
            B=self.token_budget,
            L_orig=key_states_temp.shape[-2],
            num_groups=self.num_heads // self.num_key_value_heads,
        )

    repeated_k = repeat_kv(key_states_temp, self.num_key_value_groups)
    repeated_v = repeat_kv(value_states_temp, self.num_key_value_groups)

    attn_weights_orig = torch.matmul(query_states, repeated_k.transpose(2, 3)) / math.sqrt(self.head_dim)

    mask_bottom_original = torch.tril(mask_bottom_original, diagonal=position_ids[0][0].item())

    attn_weights_orig[~mask_bottom_original] = torch.tensor(
        torch.finfo(attn_weights_orig.dtype).min,
        device=attn_weights_orig.device,
        dtype=attn_weights_orig.dtype
    )

    attn_weights_orig = nn.functional.softmax(attn_weights_orig, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights_orig, repeated_v)
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` shape mismatch. Expected {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}."
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights_orig = None

    updated_past_key_value = None
    if use_cache:
        updated_past_key_value = (
            key_states_temp,
            value_states_temp,
            new_past_k,
            new_past_v,
            old_clustered_k,
            old_clustered_v,
            old_reorder_map,
            decode_step,
            new_token_count,
            old_labels,
            cluster_centers
        )

    return attn_output, attn_weights_orig, updated_past_key_value


global layer_id
layer_id = 32


def enable_starc_attention_eval(model, args):

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_starc_attention_eval(module, args)

        global layer_id
        if isinstance(module, LlamaAttention):
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(forward, model._modules[name])

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
            model._modules[name].prefill_cluster_interval = getattr(args, "prefill_cluster_interval", 64)
            model._modules[name].prefill_chunk_size = getattr(args, "prefill_chunk_size", 16)

        elif module.__class__.__name__ in ["MistralAttention", "MistralFlashAttention2", "MistralSdpaAttention"]:
            layer_id -= 1
            m = model._modules[name]
            m.layer_id = layer_id
            m.flash_forward = m.forward
            m.forward = types.MethodType(forward, m)
            m.token_budget = args.token_budget
            m.chunk_size = args.chunk_size
            m.prefill_cluster_interval = getattr(args, "prefill_cluster_interval", 64)
            m.prefill_chunk_size = getattr(args, "prefill_chunk_size", 16)
