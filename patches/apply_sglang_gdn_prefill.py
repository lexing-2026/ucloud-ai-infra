#!/usr/bin/env python3
"""Copy-free GDN prefill (SGLang): the kernels stop writing what the next kernel only reads back.

Every kernel of the GDN prefill runs at the memory limit, so the time goes with the bytes moved:

  * Conv writes q, k, v as three contiguous segments. It wrote a packed [T, q+k+v] buffer, torch.split
    cut it into strided views and the FLA input guard copied each one with .contiguous(): three
    copies per layer, 144 in total, 17.1 ms per 8192-token prefill. The ready-made fused split stops
    at qkv_dim 8192 (this model has 10240) and would still copy.
  * Conv normalizes q and k as it goes, which replaces two l2norm launches.
  * The h kernel computes o in place, from the state and new value it already holds. It wrote h and
    v_new and chunk_fwd_o read them back: 0.51 ms per layer, 24 ms per prefill. Outputs and final
    state match the two kernels bit for bit. The only other reader of h is the prefix-cache
    snapshot, which now comes from the kernel's own TRACK_STATE path.
  * The output norm quantizes per 128 group as it goes, and out_proj takes that FP8 directly. One
    row of the norm block is one group, so the BF16 output is neither written nor read back.

The conv output feature block is one 128 wide head, so each block belongs to one of q, k, v. The
remaining tiles come from a sustained search on the same tensors (logs/iterations_b.md).

Apply with the python of the SGLang env:
  /root/infer/sg22/bin/python patches/apply_sglang_gdn_prefill.py
"""
import importlib.util
from pathlib import Path
import sys

MARK = "split_qk"

CONV = "kernels/ops/mamba/causal_conv1d_triton.py"
BACKEND = "srt/layers/attention/linear/gdn_backend.py"
BASE = "srt/layers/attention/linear/kernels/kernel_backend.py"
TRITON = "srt/layers/attention/linear/kernels/gdn_triton.py"
CHUNK = "kernels/ops/attention/fla/chunk.py"
CHUNK_H = "kernels/ops/attention/fla/chunk_delta_h.py"
WY = "kernels/ops/attention/fla/wy_fast.py"
LAYERNORM = "kernels/ops/attention/fla/layernorm_gated.py"
MODEL = "srt/models/qwen3_5.py"

EDITS = {
    CONV: [
        (
            "CAUSAL_CONV1D_FWD_BLOCK_M = 8\n",
            "CAUSAL_CONV1D_FWD_BLOCK_M = 8\n"
            "# Output feature block of the split conv: one q or k head.\n"
            "CAUSAL_CONV1D_SPLIT_BLOCK_N = 128\n",
        ),
        (
            "    USE_BLOCK_TABLE: tl.constexpr = False,\n"
            "):\n"
            "    conv_states_ptr = initial_states_ptr\n",
            "    USE_BLOCK_TABLE: tl.constexpr = False,\n"
            "    SPLIT_Q: tl.constexpr = 0,\n"
            "    SPLIT_K: tl.constexpr = 0,\n"
            "):\n"
            "    cu_seqlen = seqlen\n"
            "    conv_states_ptr = initial_states_ptr\n",
        ),
        (
            "    mask_x_1d = idx_feats < dim\n"
            "    for idx_token in range(segment_len):\n"
            "        acc = acc_preload\n",
            "    mask_x_1d = idx_feats < dim\n"
            "    if SPLIT_Q > 0:\n"
            "        # output [all q | all k | all v]: this feature block lies in one segment\n"
            "        split_feat = tl.program_id(2) * BLOCK_N\n"
            "        if split_feat < SPLIT_Q:\n"
            "            split_head = 0\n"
            "            split_width = SPLIT_Q\n"
            "        elif split_feat < SPLIT_Q + SPLIT_K:\n"
            "            split_head = SPLIT_Q\n"
            "            split_width = SPLIT_K\n"
            "        else:\n"
            "            split_head = SPLIT_Q + SPLIT_K\n"
            "            split_width = dim - SPLIT_Q - SPLIT_K\n"
            "        split_base = (\n"
            "            o_ptr + cu_seqlen.to(tl.int64) * split_head + idx_feats - split_head\n"
            "        )\n"
            "    for idx_token in range(segment_len):\n"
            "        acc = acc_preload\n",
        ),
        (
            "        if SILU_ACTIVATION:\n"
            "            acc = acc / (1 + tl.exp(-acc))\n"
            "        mask_1d = (idx_token < segment_len) & (\n"
            "            idx_feats < dim\n"
            "        )  # token-index  # feature-index\n"
            "        o_ptrs = (\n"
            "            o_ptr\n"
            "            + (sequence_start_index + token_offset + idx_token).to(tl.int64)\n"
            "            * stride_o_token\n"
            "            + (idx_feats * stride_o_dim)\n"
            "        )\n"
            "\n"
            "        tl.store(o_ptrs, acc, mask=mask_1d)\n",
            "        if SILU_ACTIVATION:\n"
            "            acc = acc / (1 + tl.exp(-acc))\n"
            "        mask_1d = (idx_token < segment_len) & (\n"
            "            idx_feats < dim\n"
            "        )  # token-index  # feature-index\n"
            "        if SPLIT_Q > 0:\n"
            "            if split_feat < SPLIT_Q + SPLIT_K:\n"
            "                # one q or k head: l2norm_fwd on the value as it would be stored\n"
            "                acc = acc.to(x_ptr.dtype.element_ty).to(tl.float32)\n"
            "                acc = acc / tl.sqrt(tl.sum(acc * acc, axis=0) + 1e-6)\n"
            "            o_ptrs = (\n"
            "                split_base\n"
            "                + (sequence_start_index + token_offset + idx_token).to(tl.int64)\n"
            "                * split_width\n"
            "            )\n"
            "        else:\n"
            "            o_ptrs = (\n"
            "                o_ptr\n"
            "                + (sequence_start_index + token_offset + idx_token).to(tl.int64)\n"
            "                * stride_o_token\n"
            "                + (idx_feats * stride_o_dim)\n"
            "            )\n"
            "\n"
            "        tl.store(o_ptrs, acc, mask=mask_1d)\n",
        ),
        (
            "    block_table: Optional[torch.Tensor] = None,\n"
            "    **kwargs,\n"
            "):\n"
            '    """support varlen + continuous batching when x is 2D tensor\n',
            "    block_table: Optional[torch.Tensor] = None,\n"
            "    split_qk: Optional[tuple] = None,\n"
            "    **kwargs,\n"
            "):\n"
            '    """support varlen + continuous batching when x is 2D tensor\n',
        ),
        (
            "    out = torch.empty_like(x)\n\n    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)\n",
            "    # split_qk=(q_dim, k_dim): out is one buffer laid out [all q | all k | all v],\n"
            "    # q and k l2-normalized per head\n"
            "    out = torch.empty_like(x) if split_qk is None else x.new_empty(x.shape)\n"
            "    split_q, split_k = split_qk or (0, 0)\n"
            "\n"
            "    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)\n",
        ),
        (
            "        BLOCK_N=256,\n"
            "        USE_BLOCK_TABLE=block_table is not None,\n"
            "        num_stages=2,\n"
            "    )\n"
            "    return out\n",
            "        BLOCK_N=256 if split_qk is None else CAUSAL_CONV1D_SPLIT_BLOCK_N,\n"
            "        USE_BLOCK_TABLE=block_table is not None,\n"
            "        SPLIT_Q=split_q,\n"
            "        SPLIT_K=split_k,\n"
            "        num_stages=2,\n"
            "    )\n"
            "    return out\n",
        ),
    ],
    BACKEND: [
        (
            "MAX_FUSED_QKV_SPLIT_DIM = 8192\n",
            "MAX_FUSED_QKV_SPLIT_DIM = 8192\n"
            "# Head width the split prefill conv can normalize in one output feature block.\n"
            "SPLIT_CONV_HEAD_DIM = 128\n",
        ),
        (
            "        conv_states: torch.Tensor,\n"
            "        cache_indices: torch.Tensor,\n"
            "    ) -> torch.Tensor:\n"
            "        block_table = self.forward_metadata.conv_block_table\n"
            '        conv_kwargs = {"block_table": block_table} if block_table is not None else {}\n'
            "        return causal_conv1d_fn(\n",
            "        conv_states: torch.Tensor,\n"
            "        cache_indices: torch.Tensor,\n"
            "        split_qk: Optional[Tuple[int, int]] = None,\n"
            "    ) -> torch.Tensor:\n"
            "        block_table = self.forward_metadata.conv_block_table\n"
            '        conv_kwargs = {"block_table": block_table} if block_table is not None else {}\n'
            "        if split_qk is not None:\n"
            "            return causal_conv1d_fn(\n"
            "                mixed_qkv.transpose(0, 1),\n"
            "                layer.conv_weights,\n"
            "                layer.bias,\n"
            "                activation=layer.activation,\n"
            "                conv_states=conv_states,\n"
            "                has_initial_state=forward_batch.extend_prefix_lens > 0,\n"
            "                cache_indices=cache_indices,\n"
            "                query_start_loc=self.forward_metadata.query_start_loc,\n"
            "                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,\n"
            "                split_qk=split_qk,\n"
            "                **conv_kwargs,\n"
            "            )\n"
            "        return causal_conv1d_fn(\n",
        ),
        (
            "        is_target_verify = forward_batch.forward_mode.is_target_verify()\n"
            "        forward_metadata = self.forward_metadata\n"
            "\n"
            "        query_start_loc = forward_metadata.query_start_loc\n"
            "        cache_indices = forward_metadata.mamba_cache_indices\n"
            "        retrieve_next_token = forward_metadata.retrieve_next_token\n",
            "        is_target_verify = forward_batch.forward_mode.is_target_verify()\n"
            "        split_conv = False\n"
            "        forward_metadata = self.forward_metadata\n"
            "\n"
            "        query_start_loc = forward_metadata.query_start_loc\n"
            "        cache_indices = forward_metadata.mamba_cache_indices\n"
            "        retrieve_next_token = forward_metadata.retrieve_next_token\n",
        ),
        (
            "            mixed_qkv = self._convolve_prefill(\n"
            "                layer,\n"
            "                forward_batch,\n"
            "                mixed_qkv,\n"
            "                conv_states_contig,\n"
            "                state_cache_indices,\n"
            "            )\n"
            "\n"
            "        actual_seq_len = mixed_qkv.shape[0]\n"
            "        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim\n"
            "        if (\n"
            "            (is_cuda() or is_hip() or is_xpu())\n",
            "            split_conv = (\n"
            "                self.kernel_dispatcher.extend_kernel.accepts_split_conv\n"
            "                and mixed_qkv.stride(0) != 1\n"
            "                and layer.head_q_dim == SPLIT_CONV_HEAD_DIM\n"
            "                and layer.head_k_dim == SPLIT_CONV_HEAD_DIM\n"
            "            )\n"
            "            mixed_qkv = self._convolve_prefill(\n"
            "                layer,\n"
            "                forward_batch,\n"
            "                mixed_qkv,\n"
            "                conv_states_contig,\n"
            "                state_cache_indices,\n"
            "                split_qk=(layer.q_dim, layer.k_dim) if split_conv else None,\n"
            "            )\n"
            "\n"
            "        actual_seq_len = mixed_qkv.shape[-1] if split_conv else mixed_qkv.shape[0]\n"
            "        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim\n"
            "        if split_conv:\n"
            "            query, key, value = torch.split(\n"
            "                mixed_qkv.reshape(-1),\n"
            "                [\n"
            "                    actual_seq_len * layer.q_dim,\n"
            "                    actual_seq_len * layer.k_dim,\n"
            "                    actual_seq_len * layer.v_dim,\n"
            "                ],\n"
            "            )\n"
            "            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)\n"
            "            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)\n"
            "            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)\n"
            "        elif (\n"
            "            (is_cuda() or is_hip() or is_xpu())\n",
        ),
        (
            "                output=kwargs.get(\"linear_attn_output\"),\n"
            "                layer_id=layer.layer_id,\n"
            "                extend_prefix_lens=forward_batch.extend_prefix_lens,\n",
            "                output=kwargs.get(\"linear_attn_output\"),\n"
            "                layer_id=layer.layer_id,\n"
            "                extend_prefix_lens=forward_batch.extend_prefix_lens,\n"
            "                qk_normalized=split_conv,\n",
        ),
    ],
    BASE: [
        (
            "    supports_track_state_snapshot: bool = False\n",
            "    supports_track_state_snapshot: bool = False\n"
            "    # True when extend() takes the q/k/v segments of the split prefill conv,\n"
            "    # with q and k already normalized (qk_normalized=True).\n"
            "    accepts_split_conv: bool = False\n",
        ),
    ],
    TRITON: [
        (
            "    supports_packed_decode: bool = not is_cpu() and not is_npu()\n"
            "    supports_strided_target_verify_qkv: bool = True\n",
            "    supports_packed_decode: bool = not is_cpu() and not is_npu()\n"
            "    supports_strided_target_verify_qkv: bool = True\n"
            "    supports_track_state_snapshot: bool = is_cuda()\n"
            "    accepts_split_conv: bool = is_cuda()\n",
        ),
        (
            "from sglang.srt.utils import is_cpu, is_npu, is_xpu\n",
            "from sglang.srt.utils import is_cpu, is_cuda, is_npu, is_xpu\n",
        ),
        (
            "        return chunk_gated_delta_rule(\n"
            "            q=q,\n"
            "            k=k,\n"
            "            v=v,\n"
            "            g=g,\n"
            "            beta=beta,\n"
            "            initial_state=recurrent_state,\n"
            "            cu_seqlens=query_start_loc,\n"
            "            head_first=False,\n"
            "            use_qk_l2norm_in_kernel=True,\n",
            "        if is_cuda():\n"
            '            inplace_update_args["track_state"] = kwargs.get("track_state")\n'
            '            inplace_update_args["track_chunk_idx"] = kwargs.get("track_chunk_idx")\n'
            "\n"
            "        return chunk_gated_delta_rule(\n"
            "            q=q,\n"
            "            k=k,\n"
            "            v=v,\n"
            "            g=g,\n"
            "            beta=beta,\n"
            "            initial_state=recurrent_state,\n"
            "            cu_seqlens=query_start_loc,\n"
            "            head_first=False,\n"
            '            use_qk_l2norm_in_kernel=not kwargs.get("qk_normalized", False),\n',
        ),
    ],
    CHUNK: [
        (
            "    chunk_indices: torch.LongTensor | None = None,\n"
            "    inplace_update: bool = True,\n"
            "):\n"
            "    g = chunk_local_cumsum(\n",
            "    chunk_indices: torch.LongTensor | None = None,\n"
            "    inplace_update: bool = True,\n"
            "    track_state: Optional[torch.Tensor] = None,\n"
            "    track_chunk_idx: Optional[torch.Tensor] = None,\n"
            "):\n"
            "    g = chunk_local_cumsum(\n",
        ),
        (
            "    h, v_new = chunk_gated_delta_rule_fwd_h(\n"
            "        k=k,\n"
            "        w=w,\n"
            "        u=u,\n"
            "        g=g,\n",
            "    if SUPPRESS_LEVEL < 3 and k.shape[-1] <= 128:\n"
            "        # o in the h kernel: h and v_new never reach memory\n"
            "        o = torch.empty_like(v)\n"
            "        chunk_gated_delta_rule_fwd_h(\n"
            "            k=k,\n"
            "            w=w,\n"
            "            u=u,\n"
            "            g=g,\n"
            "            initial_state=initial_state,\n"
            "            initial_state_indices=initial_state_indices,\n"
            "            save_new_value=False,\n"
            "            cu_seqlens=cu_seqlens,\n"
            "            chunk_indices=chunk_indices,\n"
            "            inplace_update=inplace_update,\n"
            "            track_state=track_state,\n"
            "            track_chunk_idx=track_chunk_idx,\n"
            "            q=q,\n"
            "            o=o,\n"
            "            scale=scale,\n"
            "        )\n"
            "        return g, o, A, None, None, None\n"
            "\n"
            "    assert track_state is None, \"track_state needs the fused h kernel\"\n"
            "    h, v_new = chunk_gated_delta_rule_fwd_h(\n"
            "        k=k,\n"
            "        w=w,\n"
            "        u=u,\n"
            "        g=g,\n",
        ),
        (
            "        use_qk_l2norm_in_kernel: bool = False,\n"
            "        inplace_update: bool = True,\n"
            "    ):\n"
            "        q_orig = q\n",
            "        use_qk_l2norm_in_kernel: bool = False,\n"
            "        inplace_update: bool = True,\n"
            "        track_state: Optional[torch.Tensor] = None,\n"
            "        track_chunk_idx: Optional[torch.Tensor] = None,\n"
            "    ):\n"
            "        q_orig = q\n",
        ),
        (
            "            chunk_indices=chunk_indices,\n"
            "            inplace_update=inplace_update,\n"
            "        )\n"
            "        return o.to(q.dtype), h\n",
            "            chunk_indices=chunk_indices,\n"
            "            inplace_update=inplace_update,\n"
            "            track_state=track_state,\n"
            "            track_chunk_idx=track_chunk_idx,\n"
            "        )\n"
            "        return o.to(q.dtype), h\n",
        ),
        (
            "    head_first: bool = False,\n"
            "    use_qk_l2norm_in_kernel: bool = False,\n"
            "    inplace_update: bool = True,\n"
            "):\n"
            '    r"""\n',
            "    head_first: bool = False,\n"
            "    use_qk_l2norm_in_kernel: bool = False,\n"
            "    inplace_update: bool = True,\n"
            "    track_state: Optional[torch.Tensor] = None,\n"
            "    track_chunk_idx: Optional[torch.Tensor] = None,\n"
            "):\n"
            '    r"""\n',
        ),
        (
            "        use_qk_l2norm_in_kernel,\n"
            "        inplace_update,\n"
            "    )\n"
            "    if head_first:\n",
            "        use_qk_l2norm_in_kernel,\n"
            "        inplace_update,\n"
            "        track_state,\n"
            "        track_chunk_idx,\n"
            "    )\n"
            "    if head_first:\n",
        ),
    ],
    CHUNK_H: [
        (
            '"SGLANG_GDN_CHUNK_H_BV", "32"',
            '"SGLANG_GDN_CHUNK_H_BV", "64"',
        ),
        (
            "    USE_EXP2: tl.constexpr,\n    TRACK_STATE: tl.constexpr,\n):\n",
            "    USE_EXP2: tl.constexpr,\n"
            "    TRACK_STATE: tl.constexpr,\n"
            "    q=None,\n"
            "    o=None,\n"
            "    scale=None,\n"
            "    FUSE_O: tl.constexpr = False,\n"
            "):\n",
        ),
        (
            "    h += ((boh * H + i_h) * V * K).to(tl.int64)\n",
            "    if FUSE_O:\n"
            "        q += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)\n"
            "        o += ((bos * H + i_h) * V).to(tl.int64)\n"
            "    else:\n"
            "        h += ((boh * H + i_h) * V * K).to(tl.int64)\n",
        ),
        (
            "        last_idx = min((i_t + 1) * BT, T) - 1\n"
            "        if USE_G:\n"
            "            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)\n",
            "        if FUSE_O:\n"
            "            # chunk_fwd_kernel_o on the state and new value this program holds\n"
            "            b_q = tl.load(\n"
            "                tl.make_block_ptr(\n"
            "                    q, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0)\n"
            "                ),\n"
            "                boundary_check=(0, 1),\n"
            "            )\n"
            "            b_o = tl.dot(b_q, tl.trans(b_h1).to(b_q.dtype))\n"
            "            b_A = tl.dot(\n"
            "                b_q,\n"
            "                tl.load(\n"
            "                    tl.make_block_ptr(\n"
            "                        k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)\n"
            "                    ),\n"
            "                    boundary_check=(0, 1),\n"
            "                ),\n"
            "            )\n"
            "            if K > 64:\n"
            "                b_q = tl.load(\n"
            "                    tl.make_block_ptr(\n"
            "                        q, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0)\n"
            "                    ),\n"
            "                    boundary_check=(0, 1),\n"
            "                )\n"
            "                b_o += tl.dot(b_q, tl.trans(b_h2).to(b_q.dtype))\n"
            "                b_A += tl.dot(\n"
            "                    b_q,\n"
            "                    tl.load(\n"
            "                        tl.make_block_ptr(\n"
            "                            k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)\n"
            "                        ),\n"
            "                        boundary_check=(0, 1),\n"
            "                    ),\n"
            "                )\n"
            "            if USE_G:\n"
            "                b_gc = tl.load(\n"
            "                    tl.make_block_ptr(\n"
            "                        g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)\n"
            "                    ),\n"
            "                    boundary_check=(0,),\n"
            "                )\n"
            "                b_o = b_o * exp(b_gc)[:, None]\n"
            "                b_A = b_A * safe_exp(b_gc[:, None] - b_gc[None, :])\n"
            "            o_i = tl.arange(0, BT)\n"
            "            b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0)\n"
            "            b_vn = b_v.to(k.dtype.element_ty)\n"
            "            b_o = b_o * scale + tl.dot(b_A.to(b_vn.dtype), b_vn) * scale\n"
            "            p_o = tl.make_block_ptr(\n"
            "                o, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)\n"
            "            )\n"
            "            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))\n"
            "\n"
            "        last_idx = min((i_t + 1) * BT, T) - 1\n"
            "        if USE_G:\n"
            "            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)\n",
        ),
        (
            "    chunk_offsets: Optional[torch.Tensor] = None,\n"
            ") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n",
            "    chunk_offsets: Optional[torch.Tensor] = None,\n"
            "    q: Optional[torch.Tensor] = None,\n"
            "    o: Optional[torch.Tensor] = None,\n"
            "    scale: Optional[float] = None,\n"
            ") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n",
        ),
        (
            "    h = k.new_empty(B, NT, H, V, K)\n"
            "\n"
            "    v_new = torch.empty_like(u) if save_new_value else None\n",
            "    h = k.new_empty(B, NT, H, V, K) if o is None else None\n"
            "\n"
            "    v_new = torch.empty_like(u) if save_new_value and o is None else None\n",
        ),
        (
            "        TRACK_STATE=track_state is not None,\n"
            "    )\n"
            "    return h, v_new\n",
            "        TRACK_STATE=track_state is not None,\n"
            "        q=q,\n"
            "        o=o,\n"
            "        scale=scale,\n"
            "        FUSE_O=o is not None,\n"
            "    )\n"
            "    return h, v_new\n",
        ),
    ],
    LAYERNORM: [
        (
            "    Q=None,\n    S=None,\n    QUANT: tl.constexpr = False,\n):\n",
            "    Q=None,\n"
            "    S=None,\n"
            "    QUANT: tl.constexpr = False,\n"
            "    GROUP_QUANT: tl.constexpr = False,\n"
            "):\n",
        ),
        (
            "        tl.store(S + tl.program_id(0), y_s)\n"
            "    else:\n"
            "        # Write output\n",
            "        tl.store(S + tl.program_id(0), y_s)\n"
            "    elif GROUP_QUANT:  # each row is one group: per_token_group_quant of the stored value\n"
            "        y = tl.where(mask, y.to(Y.dtype.element_ty).to(tl.float32), 0.0)\n"
            "        y_amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10)\n"
            "        y_s = y_amax / 448.0\n"
            "        y_q = tl.clamp(y * (448.0 / y_amax)[:, None], -448.0, 448.0)\n"
            "        y_q = y_q.to(Q.dtype.element_ty)\n"
            "        tl.store(Q + rows[:, None] * N + col_offsets, y_q, mask=mask)\n"
            "        tl.store(S + rows, y_s, mask=rows < M)\n"
            "    else:\n"
            "        # Write output\n",
        ),
        (
            '    activation: str = "swish",\n'
            "    quant_heads: int = 0,\n"
            "):\n",
            '    activation: str = "swish",\n'
            "    quant_heads: int = 0,\n"
            "    group_quant: bool = False,\n"
            "):\n",
        ),
        (
            '        quant = {"Q": q, "S": s, "QUANT": True}\n',
            '        quant = {"Q": q, "S": s, "QUANT": True}\n'
            "    if group_quant:\n"
            "        assert ngroups == 1\n"
            "        q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)\n"
            "        s = torch.empty((M,), dtype=torch.float32, device=x.device)\n"
            '        quant = {"Q": q, "S": s, "GROUP_QUANT": True}\n',
        ),
        (
            "    return ((q, s) if quant_heads else out), mean, rstd\n",
            "    return ((q, s) if quant_heads or group_quant else out), mean, rstd\n",
        ),
    ],
    MODEL: [
        (
            "            return self.out_proj(fp8)[0]\n"
            "\n"
            "        core_attn_out = self.norm(core_attn_out, z)\n",
            "            return self.out_proj(fp8)[0]\n"
            "\n"
            '        if use_strided_prefill_z and hasattr(self.out_proj.weight, "tiles"):\n'
            "            (fp8, scale), _, _ = _layer_norm_fwd(\n"
            "                core_attn_out,\n"
            "                self.norm.weight,\n"
            "                None,\n"
            "                self.norm.eps,\n"
            "                z=z,\n"
            "                is_rms_norm=True,\n"
            "                activation=self.norm.activation,\n"
            "                group_quant=True,\n"
            "            )\n"
            "            token_dim = nv * core_attn_out.shape[-1]\n"
            "            return self.out_proj((fp8.view(-1, token_dim), scale.view(-1, nv)))[0]\n"
            "\n"
            "        core_attn_out = self.norm(core_attn_out, z)\n",
        ),
    ],
    WY: [
        (
            "    BK = 64\n    BV = 64\n",
            "    BK = 128\n    BV = 128\n",
        ),
        (
            "        IS_VARLEN=cu_seqlens is not None,\n"
            "        num_warps=4,\n"
            "        num_stages=3,\n",
            "        IS_VARLEN=cu_seqlens is not None,\n"
            "        num_warps=4,\n"
            "        num_stages=2,\n",
        ),
    ],
}


def guard_h_stores(text):
    """The per-chunk h stores only run outside the fused kernel."""
    start = text.index("        p_h1 = tl.make_block_ptr(\n            h + i_t * stride_h")
    end = text.index("        if TRACK_STATE and i_t == i_track:\n")
    block = "".join("    " + line if line.strip() else line for line in text[start:end].splitlines(True))
    return text[:start] + "        if not FUSE_O:\n" + block + text[end:]


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    root = Path(spec.origin).parent
    if MARK in (root / CONV).read_text():
        print(f"already patched: {root}")
        return 0
    patched = {}
    for name, edits in EDITS.items():
        text = (root / name).read_text()
        for old, new in edits:
            if new in text:
                continue
            if text.count(old) != 1:
                print(f"expected one match in {name}:\n{old}", file=sys.stderr)
                return 1
            text = text.replace(old, new)
        patched[name] = guard_h_stores(text) if name == CHUNK_H else text
    for name, text in patched.items():
        (root / name).write_text(text)
        print(f"patched: {root / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
