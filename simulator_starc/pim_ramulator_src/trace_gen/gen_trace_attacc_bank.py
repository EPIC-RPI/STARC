import argparse
import math
import copy
import numpy as np

# Row-wise sparsity configuration (row = blkrow = 16 tokens by default)
ROW_SIZE = 16  # tokens per row
SPARSITY_RATIO = 0.25  # keep this fraction of rows for each new token
TOKENS_PER_ROW = ROW_SIZE
KV_BUDGET = 0  # 0 => ratio-based; >0 => fixed KV budget (pairs)

def calculate_page_config(L):
    """Ratio-based active row selection."""
    total_pages = math.ceil(L / ROW_SIZE)
    active_pages = max(1, math.ceil(total_pages * SPARSITY_RATIO))
    return total_pages, active_pages

def calculate_page_config_fixed_budget(L, kv_budget):
    """Fixed-budget active row selection (kv_budget in KV pairs)."""
    total_pages = math.ceil(L / ROW_SIZE)
    row_budget = math.ceil(kv_budget / ROW_SIZE)
    active_pages = min(row_budget, total_pages)
    return total_pages, active_pages



# Model / hardware parameters

dhead = 128
max_L = 2048
data_size = 16  # FP16 nominal; overridden by CLI

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32  # bytes
n_mac = 16

# Address granularities
HBM_GS = {}
HBM_GS['col'] = prefetch_size
HBM_GS['row'] = n_col * HBM_GS['col']
HBM_GS['ba'] = n_row * HBM_GS['row']
HBM_GS['bg'] = n_bank * HBM_GS['ba']
HBM_GS['rank'] = n_bg * HBM_GS['bg']
HBM_GS['pch'] = n_rank * HBM_GS['rank']
HBM_GS['ch'] = n_pch * HBM_GS['pch']
HBM_GS['hbm'] = n_channel * HBM_GS['ch']
HBM_GS['attacc'] = max_n_hbm * HBM_GS['hbm']

# Trace buffers
cmd_score_wrgb = []
cmd_score_mac = []
cmd_score_mvsb = []
cmd_sfm = []
cmd_context_mvgb = []
cmd_context_mac = []
cmd_context_mvsb = []

cmd_norm_mvgb = []
cmd_norm_mac = []
cmd_norm_mvsb = []
cmd_cos_mvgb = []
cmd_cos_mac = []
cmd_cos_mvsb = []

valid_channels = []

def cmd_list_reset():
    """Reset all per-run trace buffers."""
    global cmd_score_wrgb, cmd_score_mac, cmd_score_mvsb, cmd_sfm
    global cmd_context_mvgb, cmd_context_mac, cmd_context_mvsb, valid_channels

    cmd_score_wrgb.clear()
    cmd_score_mac.clear()
    cmd_score_mvsb.clear()
    cmd_sfm.clear()
    cmd_context_mvgb.clear()
    cmd_context_mac.clear()
    cmd_context_mvsb.clear()
    valid_channels.clear()
    cmd_norm_mvgb.clear()
    cmd_norm_mac.clear()
    cmd_norm_mvsb.clear()
    cmd_cos_mvgb.clear()
    cmd_cos_mac.clear()
    cmd_cos_mvsb.clear()


def Attention(L, key_addr, val_addr, itr, valid_channel=n_channel,
              sparsity=False, iter_num=0, cluster_width=64, norm_addr=None,
              cluater_itr=0, add_cluster=False):
    """Generate AttAcc-like trace."""
    cmd_score_wrgb.append([])
    cmd_score_mac.append([])
    cmd_score_mvsb.append([])
    cmd_sfm.append([])
    cmd_context_mvgb.append([])
    cmd_context_mac.append([])
    cmd_context_mvsb.append([])

    cmd_norm_mvgb.append([])
    cmd_norm_mac.append([])
    cmd_norm_mvsb.append([])
    cmd_cos_mvgb.append([])
    cmd_cos_mac.append([])
    cmd_cos_mvsb.append([])

    valid_channels.append(valid_channel)

    # Effective length under row-sparsity
    if sparsity:
        if KV_BUDGET > 0:
            total_pages, active_pages = calculate_page_config_fixed_budget(L, KV_BUDGET)
        else:
            total_pages, active_pages = calculate_page_config(L)
        effective_L = active_pages * ROW_SIZE
    else:
        effective_L = L

    def cluster_norm_squre(addr_offset, L, cluster_width=64):
        for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):
            cmd_norm_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
                idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_norm_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            # MVSB can be aggregated at blkrow=16
            if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
                cmd_norm_mvsb[itr].append([])
                for bg_idx in range(n_bg):
                    for rank in range(n_rank):
                        for lch in range(math.ceil(valid_channel)):
                            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                        bg_idx * HBM_GS['bg']
                            hex_addr = hex(bank_addr)[2:]
                            cmd_norm_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def cluater_norm_divide(addr_offset, L, cluster_width=64):
        for rank in range(n_rank):
            for bg_idx in range(n_bg):
                for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                               bg_idx * HBM_GS['bg'] + col_idx
                        hex_addr = hex(addr)[2:]
                        cmd_norm_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

        for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):
            cmd_norm_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
                idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_norm_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

    def cluster_similar(addr_offset, L, cluster_width=64):
        for clu_idx in range(cluater_itr):
            for n_idx in range(math.ceil(L * L / cluster_width / n_pch / n_rank / n_bg)):
                cmd_cos_mac[itr].append([])
                for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
                    idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                        hex_addr = hex(addr)[2:]
                        cmd_cos_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

                if n_idx % 16 == 15 or n_idx == math.ceil(L * L / cluster_width / n_pch / n_rank / n_bg) - 1:
                    cmd_cos_mvsb[itr].append([])
                    for bg_idx in range(n_bg):
                        for rank in range(n_rank):
                            for lch in range(math.ceil(valid_channel)):
                                bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                            bg_idx * HBM_GS['bg']
                                hex_addr = hex(bank_addr)[2:]
                                cmd_cos_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def cluster_update(addr_offset, L, cluster_width=64):
        K = math.ceil(L / cluster_width)

        for it in range(cluater_itr):
            for rank in range(n_rank):
                for bg_idx in range(n_bg):
                    for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
                        for lch in range(math.ceil(valid_channel)):
                            addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                   bg_idx * HBM_GS['bg'] + col_idx
                            hex_addr = hex(addr)[2:]
                            cmd_cos_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

            # Write back updated centroids
            centroid_base = addr_offset + (1 << 22)
            n_centroid_rows = math.ceil(K / ROW_SIZE)
            cols_per_vec = math.ceil(dhead / (n_bank * n_mac))

            for row_idx in range(n_centroid_rows):
                row_off = row_idx * HBM_GS['row']
                for ba_idx in range(n_bank):
                    for col_idx in range(cols_per_vec):
                        for lch in range(math.ceil(valid_channel)):
                            dst = centroid_base + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + row_off + col_idx
                            hex_dst = hex(dst)[2:]
                            cmd_cos_mvgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_dst))

    def cluster_remap(key_base, val_base, key_new_base, val_new_base, L):
        """KV remap"""
        n_rows = math.ceil(L / ROW_SIZE)
        cols_per_vec = math.ceil(dhead / (n_bank * n_mac))

        for row_idx in range(n_rows):
            row_off = row_idx * HBM_GS['row']

            for ba_idx in range(n_bank):
                for col_idx in range(cols_per_vec):
                    for lch in range(math.ceil(valid_channel)):
                        src = key_base + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + row_off + col_idx
                        dst = key_new_base + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + row_off + col_idx
                        cmd_cos_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex(src)[2:]))
                        cmd_cos_mvgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex(dst)[2:]))

            for ba_idx in range(n_bank):
                for col_idx in range(cols_per_vec):
                    for lch in range(math.ceil(valid_channel)):
                        src = val_base + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + row_off + col_idx
                        dst = val_new_base + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + row_off + col_idx
                        cmd_cos_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex(src)[2:]))
                        cmd_cos_mvgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex(dst)[2:]))

    def score_cpvec(addr_offset, L):
        for ba_idx in range(n_bank):
            for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
                    hex_addr = hex(addr)[2:]
                    cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def score_mac(addr_offset, L):
        for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):
            cmd_score_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
                idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
                cmd_score_mvsb[itr].append([])
                for bg_idx in range(n_bg):
                    for rank in range(n_rank):
                        for lch in range(math.ceil(valid_channel)):
                            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                        bg_idx * HBM_GS['bg']
                            hex_addr = hex(bank_addr)[2:]
                            cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def context_cpvec(addr_offset, L):
        for rank in range(n_rank):
            for bg_idx in range(n_bg):
                for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                               bg_idx * HBM_GS['bg'] + col_idx
                        hex_addr = hex(addr)[2:]
                        cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    def context_mac(addr_offset, L):
        for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
            cmd_context_mac[itr].append([])
            for k_idx in range(math.ceil(L / (n_pch * n_rank * n_bg))):
                idx = k_idx + n_idx * math.ceil(L / (n_pch * n_rank * n_bg))
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            cmd_context_mvsb[itr].append([])
            for ba_idx in range(n_bank):
                for rank in range(n_rank):
                    for lch in range(math.ceil(valid_channel)):
                        bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                    ba_idx * HBM_GS['ba']
                        hex_addr = hex(bank_addr)[2:]
                        cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def softmax(L):
        for lch in range(math.ceil(valid_channel)):
            addr = lch * HBM_GS['ch']
            hex_addr = hex(addr)[2:]
            cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

    if add_cluster:
        cluster_norm_squre(norm_addr, L, cluster_width)
        cluater_norm_divide(norm_addr, L, cluster_width)
        cluster_similar(norm_addr, L, cluster_width)
        cluster_update(norm_addr, L, cluster_width)

        # New address space for remapped KV (modeled)
        key_new_addr = norm_addr + (1 << 23)
        val_new_addr = key_new_addr + (1 << 23)

        cluster_remap(key_addr, val_addr, key_new_addr, val_new_addr, L)

    score_cpvec(key_addr, effective_L)
    score_mac(key_addr, effective_L)
    softmax(effective_L)
    context_cpvec(val_addr, effective_L)
    context_mac(val_addr, effective_L)


def run_attention(dhead, n_head_per_hbm, L, trace_file_name, sparsity=False, cluster_width=64, add_cluster=False):
    def mac_iters(lst, stride=16):
        return math.ceil(len(lst) / stride) if lst else 0

    partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
    v_offset = pow(2, 23)
    norm_offset = pow(2, 23)

    cmd_list_reset()
    num_itr = math.ceil(n_head_per_hbm / n_channel)
    for itr in range(num_itr):
        remainder = 0
        if (n_head_per_hbm / ((itr + 1) * n_channel) < 1):
            remainder = n_head_per_hbm % n_channel
        key_addr = itr * partition_size
        val_addr = key_addr + v_offset
        norm_addr = val_addr + norm_offset

        if remainder == 0:
            Attention(L, key_addr, val_addr, itr, n_channel, sparsity, itr,
                      cluster_width=cluster_width, norm_addr=norm_addr, cluater_itr=16, add_cluster=add_cluster)
        else:
            Attention(L, key_addr, val_addr, itr, remainder, sparsity, itr,
                      cluster_width=cluster_width, norm_addr=norm_addr, cluater_itr=16, add_cluster=add_cluster)

    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

    total_cmd = []

    # Cluster trace scheduling
    if add_cluster:
        if sparsity:
            for i in range(0, num_itr - 1, 2):
                total_cmd += barrier
                length = math.ceil(L / n_pch / n_rank / n_bg / 16)
                for j in range(0, length + 1):
                    if not j == length:
                        stride = 16
                        for k in range(stride):
                            if (j * stride + k) >= len(cmd_norm_mac[i]):
                                break
                            total_cmd += cmd_norm_mac[i][j * stride + k]
                    if not j == 0:
                        total_cmd += cmd_norm_mvsb[i][j - 1]
                    if not j == length:
                        stride = int(n_bank * math.ceil(dhead / n_bank / n_mac) * math.ceil(valid_channels[i + 1]) / length)
                        for k in range(stride):
                            if (j * stride + k) >= len(cmd_norm_mvgb[i + 1]):
                                break
                            total_cmd.append(cmd_norm_mvgb[i + 1][j * stride + k])
                    if not j == length:
                        total_cmd += barrier

                length = math.ceil(L / n_pch / n_rank / n_bg / 16)
                for j in range(0, length + 1):
                    if not j == length:
                        stride = 16
                        for k in range(stride):
                            if (j * stride + k) >= len(cmd_norm_mac[i + 1]):
                                break
                            total_cmd += cmd_norm_mac[i + 1][j * stride + k]
                    if not j == 0:
                        total_cmd += cmd_norm_mvsb[i + 1][j - 1]
                    if not j == length:
                        if j >= math.floor(length / 2):
                            stride = int(
                                n_rank * n_bg *
                                math.ceil(L * L / cluster_width / (n_pch * n_rank * n_bg * n_mac)) *
                                math.ceil(valid_channels[i]) / math.ceil(length / 2)
                            )
                            for k in range(stride):
                                if ((j - math.floor(length / 2)) * stride + k) >= len(cmd_cos_mvgb[i]):
                                    break
                                total_cmd.append(cmd_cos_mvgb[i][(j - math.floor(length / 2)) * stride + k])
                    if not j == length:
                        total_cmd += barrier

                length = math.ceil(L * L / cluster_width / n_pch / n_rank / n_bg / 16)
                for j in range(0, length + 1):
                    if not j == length:
                        total_cmd += cmd_cos_mac[i][j]
                    if not j == 0:
                        total_cmd += cmd_cos_mvgb[i][j - 1]
                    if not j == length:
                        total_cmd += barrier

            if num_itr % 2 != 0:
                i = num_itr - 1
                total_cmd += cmd_norm_mvgb[i]
                total_cmd += barrier

                length = math.ceil(L / n_pch / n_rank / n_bg / 16)
                for j in range(0, length + 1):
                    if not j == length:
                        stride = 16
                        for k in range(stride):
                            if (j * stride + k) >= len(cmd_norm_mac[i]):
                                break
                            total_cmd += cmd_norm_mac[i][j * stride + k]
                    if not j == 0:
                        total_cmd += cmd_norm_mvsb[i][j - 1]
                    if not j == length:
                        total_cmd += barrier

                total_cmd += cmd_cos_mvgb[i]
                total_cmd += barrier

                length = math.ceil(L * L / cluster_width / n_bank / n_mac / 16)
                for j in range(0, length + 1):
                    if not j == length:
                        total_cmd += cmd_cos_mac[i][j]
                    if not j == 0:
                        total_cmd += cmd_cos_mvsb[i][j - 1]
                    if not j == length:
                        total_cmd += barrier

    # Main overlap schedule
    for i in range(0, num_itr - 1, 2):
        total_cmd += cmd_score_wrgb[i]

        if i == 0:
            for j in range(valid_channels[i]):
                total_cmd.append(cmd_score_mac[i][0][j])

        total_cmd += barrier

        mac0_stride = 16
        mac0_iters = mac_iters(cmd_score_mac[i], mac0_stride)
        mvsb0_len = len(cmd_score_mvsb[i])
        length = max(mac0_iters, mvsb0_len)

        for j in range(0, length + 1):
            if j != length:
                for k in range(mac0_stride):
                    idx = j * mac0_stride + k
                    if idx >= len(cmd_score_mac[i]):
                        break
                    total_cmd += cmd_score_mac[i][idx]

            if j != 0 and (j - 1) < len(cmd_score_mvsb[i]):
                total_cmd += cmd_score_mvsb[i][j - 1]

            if j != length:
                wrgb1_stride = int(
                    n_bank * math.ceil(dhead / n_bank / n_mac) * math.ceil(valid_channels[i + 1]) / max(1, length)
                )
                for k in range(wrgb1_stride):
                    idx = j * wrgb1_stride + k
                    if idx >= len(cmd_score_wrgb[i + 1]):
                        break
                    total_cmd.append(cmd_score_wrgb[i + 1][idx])

            if j != length:
                total_cmd += barrier

        mac1_stride = 16
        mac1_iters = mac_iters(cmd_score_mac[i + 1], mac1_stride)
        mvsb1_len = len(cmd_score_mvsb[i + 1])
        mvgb0_len = len(cmd_context_mvgb[i])
        length2 = max(mac1_iters, mvsb1_len)

        for j in range(0, length2 + 1):
            if j != length2:
                for k in range(mac1_stride):
                    idx = j * mac1_stride + k
                    if idx >= len(cmd_score_mac[i + 1]):
                        break
                    total_cmd += cmd_score_mac[i + 1][idx]

            if j != 0 and (j - 1) < len(cmd_score_mvsb[i + 1]):
                total_cmd += cmd_score_mvsb[i + 1][j - 1]

            if j == 0:
                total_cmd += cmd_sfm[i]

            if j != length2 and j >= math.floor(length2 / 2):
                mvgb_stride0 = int(
                    n_rank * n_bg * math.ceil(L / (n_pch * n_rank * n_bg * n_mac)) * math.ceil(valid_channels[i]) /
                    max(1, math.ceil(length2 / 2))
                )
                base = (j - math.floor(length2 / 2)) * mvgb_stride0
                for k in range(mvgb_stride0):
                    idx = base + k
                    if idx >= mvgb0_len:
                        break
                    total_cmd.append(cmd_context_mvgb[i][idx])

            if j != length2:
                total_cmd += barrier

        ctx_mac0_len = len(cmd_context_mac[i])
        ctx_mvsb0_len = len(cmd_context_mvsb[i])
        mvgb1_len = len(cmd_context_mvgb[i + 1])
        length3 = max(ctx_mac0_len, ctx_mvsb0_len)

        for j in range(0, length3 + 1):
            if j != length3 and j < ctx_mac0_len:
                total_cmd += cmd_context_mac[i][j]

            if j != 0 and (j - 1) < ctx_mvsb0_len:
                total_cmd += cmd_context_mvsb[i][j - 1]

            if j == 0:
                total_cmd += cmd_sfm[i + 1]

            if j != length3 and j >= math.floor(length3 / 2):
                mvgb_stride1 = int(
                    n_rank * n_bg * math.ceil(L / (n_pch * n_rank * n_bg * n_mac)) * math.ceil(valid_channels[i + 1]) /
                    max(1, math.ceil(length3 / 2))
                )
                base = (j - math.floor(length3 / 2)) * mvgb_stride1
                for k in range(mvgb_stride1):
                    idx = base + k
                    if idx >= mvgb1_len:
                        break
                    total_cmd.append(cmd_context_mvgb[i + 1][idx])

            if j != length3:
                total_cmd += barrier

        ctx_mac1_len = len(cmd_context_mac[i])
        ctx_mvsb1_len = len(cmd_context_mvsb[i])
        length4 = max(ctx_mac1_len, ctx_mvsb1_len)

        for j in range(0, length4 + 1):
            if j != length4 and j < ctx_mac1_len:
                total_cmd += cmd_context_mac[i][j]

            if j != 0 and (j - 1) < ctx_mvsb1_len:
                total_cmd += cmd_context_mvsb[i][j - 1]

            if j != length4:
                total_cmd += barrier

    if num_itr % 2 != 0:
        i = num_itr - 1
        total_cmd += cmd_score_wrgb[i]
        total_cmd += barrier

        mac_tail_stride = 16
        mac_tail_iters = mac_iters(cmd_score_mac[i], mac_tail_stride)
        mvsb_tail_len = len(cmd_score_mvsb[i])
        length_tail = max(mac_tail_iters, mvsb_tail_len)

        for j in range(0, length_tail + 1):
            if j != length_tail:
                for k in range(mac_tail_stride):
                    idx = j * mac_tail_stride + k
                    if idx >= len(cmd_score_mac[i]):
                        break
                    total_cmd += cmd_score_mac[i][idx]

            if j != 0 and (j - 1) < len(cmd_score_mvsb[i]):
                total_cmd += cmd_score_mvsb[i][j - 1]

            if j != length_tail:
                total_cmd += barrier

        total_cmd += cmd_sfm[i]
        total_cmd += cmd_context_mvgb[i]
        total_cmd += barrier

        ctx_mac_tail_len = len(cmd_context_mac[i])
        ctx_mvsb_tail_len = len(cmd_context_mvsb[i])
        length_ctx_tail = max(ctx_mac_tail_len, ctx_mvsb_tail_len)

        for j in range(0, length_ctx_tail + 1):
            if j != length_ctx_tail and j < ctx_mac_tail_len:
                total_cmd += cmd_context_mac[i][j]

            if j != 0 and (j - 1) < ctx_mvsb_tail_len:
                total_cmd += cmd_context_mvsb[i][j - 1]

            if j != length_ctx_tail:
                total_cmd += barrier

    with open(trace_file_name, 'w') as trace_file:
        for cmd in total_cmd:
            trace_file.write(cmd + "\n")


def main():
    global dhead, max_L, data_size, n_mac
    global ROW_SIZE, SPARSITY_RATIO, KV_BUDGET

    parser = argparse.ArgumentParser(
        description="Generate bank-level AttAcc trace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-dh", "--dhead", type=int, default=128, help="Head dimension")
    parser.add_argument("-nh", "--nhead", type=int, default=64, help="Number of heads per HBM")
    parser.add_argument("-l", "--seqlen", type=int, default=2048, help="Sequence length L")
    parser.add_argument("-maxl", "--maxlen", type=int, default=409600, help="Maximum L for address partitioning")
    parser.add_argument("-db", "--dbyte", type=int, default=2, help="Data bytes per element (e.g., 2 for FP16)")
    parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace", help="Output trace path")

    parser.add_argument("-sp", "--sparsity", action="store_true", help="Enable row-sparsity (subset of rows)")
    parser.add_argument("-rs", "--row_size", type=int, default=16, help="Tokens per row (blkrow)")
    parser.add_argument("-sr", "--sparsity_ratio", type=float, default=0.25, help="Fraction of rows kept if KV budget is 0")
    parser.add_argument("--kv_budget", type=int, default=1024, help="Fixed KV budget (pairs); 0 => ratio-based")


    parser.add_argument("--add_cluster", action="store_true", help="Include clustering/remap overhead")

    args = parser.parse_args()
    dhead = args.dhead
    max_L = args.maxlen
    L = args.seqlen
    n_head_per_hbm = args.nhead

    data_size = args.dbyte
    n_mac = int(HBM_GS['col'] / data_size)

    if args.sparsity:
        ROW_SIZE = args.row_size
        SPARSITY_RATIO = args.sparsity_ratio
        KV_BUDGET = args.kv_budget

    print("------   Make a trace of bank-level AttAcc   ------")

    total_pages = math.ceil(L / ROW_SIZE) if args.sparsity else 0
    if args.sparsity:
        if args.kv_budget > 0:
            _, active_pages = calculate_page_config_fixed_budget(L, args.kv_budget)
        else:
            _, active_pages = calculate_page_config(L)

        print(" Row-sparsity mode enabled")


    print(" Configuration Arguments:")
    for key, value in vars(args).items():
        print(f"     {key}: {value}")
    print("---------------------------------------------------")

    run_attention(dhead, n_head_per_hbm, L, args.output, args.sparsity, cluster_width=64, add_cluster=args.add_cluster)


if __name__ == "__main__":
    main()
