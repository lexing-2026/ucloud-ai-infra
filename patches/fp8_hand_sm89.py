"""Hand-scheduled block-FP8 GEMM for sm_89 (RTX 4090). Installed by patches/apply_sglang_fp8_hand_gemm.py.

With SGLANG_FP8_HAND_GEMM=1 every FP8 linear except lm_head moves to tile first storage right after
loading and only these kernels read it:

  W8[block, tile, row, k]  u8, 16 KiB per (block, tile), so a B copy is sequential lines in one DRAM
  page. Row major storage turns the same copy into 128 stridden 128 B reads.

Decode is limited by HBM bandwidth and this layout reads 930-950 GB/s where the tuned Triton tiles
read 855-895. Prefill sits at the 450 W cap, so what counts there is energy per MAC, not only tensor
core occupancy:

  * BM=128 shares one B tile between twice the rows: 33% less on-chip traffic per MAC and about
    100 MHz more clock at the same power.
  * Persistent CTAs walk NB blocks each. ptxas -O1 keeps the written schedule; -O3 undoes it.
  * FP32 scaling is software-pipelined into the mma stream of the last K32 step and needs 72 FP
    ops per warp and tile instead of 128 (s[i] = a_s[i] * b_s is computed once).
  * Scales travel without a second barrier or an SMEM broadcast per warp.
  * A registers are reloaded in place right after their last reader.

Lane mapping rules that are not optional (each one cost a factor when ignored):
  * one cp.async covers 4 cache lines, 8 lanes each, so lanes coalesce into 4 L2 requests;
  * SMEM uses K_SW128: chunk ^= row & 7, otherwise every ldmatrix phase is an 8-way bank conflict.

Results differ from the Triton kernel only in FP32 rounding of the scale product
(mean relative difference about 1e-7), so the serving config has to pass IFBench again.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)

ENABLED = os.getenv("SGLANG_FP8_HAND_GEMM", "0") == "1"
FUSED_ENABLED = ENABLED and os.getenv("SGLANG_FP8_HAND_FUSED", "1") == "1"
FUSED_MIN_M = int(os.getenv("SGLANG_FP8_HAND_FUSED_MIN_M", "1024"))

# Weight K of Qwen3.8-27B: QKV/gate-up, down, attention out.
KS = (5120, 6144, 17408)
FUSED_K = 5120
BK = 128
B_STAGE = 0x4000
S_STAGE = 0x400
# Picked by the search in logs/iterations_b.md (sustained ms at the power cap).
GAP, CPS, OPT = 5, 8, 1
PREFILL = dict(BM=128, stages=2, NB=4, bpos=5, G=32)
# Decode tiles by weight size class, from the same log: (largest M, BM, stages, NB, bpos).
DECODE_LARGE = ((16, 16, 2, 2, 0), (32, 32, 2, 2, 5), (128, 32, 3, 1, 5))
DECODE_MID = ((16, 16, 2, 1, 0), (128, 32, 3, 1, 0))
DECODE_SMALL = ((32, 16, 3, 1, 0), (128, 32, 3, 1, 5))
CACHE_ATTR_PREFER_SHARED = 1
MAX_DYNAMIC_SHARED_ATTR = 8


def threads(BM):
    return 256 if BM == 128 else 128


def stage_sizes(BM):
    return BM * 128, B_STAGE, S_STAGE


def smem_bytes(stages, BM):
    return stages * sum(stage_sizes(BM))


def select(N, M):
    """Tile of an M x N product."""
    if M > 128:
        return PREFILL
    rows = DECODE_LARGE if N >= 30000 else DECODE_MID if N >= 10000 else DECODE_SMALL
    for top, BM, stages, NB, bpos in rows:
        if M <= top:
            return dict(BM=BM, stages=stages, NB=NB, bpos=bpos, G=1024)


def _configs(K):
    groups = (DECODE_LARGE, DECODE_MID, DECODE_SMALL) if K == FUSED_K else (DECODE_SMALL,)
    rows = {row[1:] for group in groups for row in group}
    return [PREFILL] + [dict(BM=BM, stages=stages, NB=NB, bpos=bpos, G=1024) for BM, stages, NB, bpos in sorted(rows)]


def gen(K, stages=2, gap=5, NB=4, cps=8, bpos=5, name="fp8_gemm_hand", BM=128):
    S = stages
    NT = threads(BM)
    LOGM = {16: 4, 32: 5, 64: 6, 128: 7}[BM]
    AF = min(BM, 64) // 16
    NF = AF * 4
    gap = min(gap, NF - 1)
    bpos = min(bpos, NF - 4)
    A_STAGE, _, _ = stage_sizes(BM)
    SB0 = S * A_STAGE
    SS0 = SB0 + S * B_STAGE
    L = []
    e = L.append
    e(".version 8.5\n.target sm_89\n.address_size 64\n.extern .shared .align 16 .b8 smem[];\n")
    e(f".visible .entry {name}(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pAs, .param .u64 pBs,")
    e("  .param .u32 pM, .param .u32 pN, .param .u32 pG)")
    e(f".reqntid {NT}\n{{")
    e("  .reg .b32 %T<64>;\n  .reg .b32 %C<64>;\n  .reg .b32 %A<16>;\n  .reg .b32 %B<32>;\n  .reg .b32 %S<8>;")
    e("  .reg .b32 %x<16>;\n  .reg .b64 %d<10>;\n  .reg .pred %p<8>;\n  .reg .pred %pr<4>;\n  .reg .pred %pq<4>;\n  .reg .pred %pw, %pz;")
    e("  .reg .b32 %la<4>;\n  .reg .b32 %lb<2>;\n  .reg .b32 %ra<4>;\n  .reg .b32 %rb<2>;\n  .reg .b32 %da<2>;\n  .reg .b32 %db<2>;")
    e("  .reg .b32 %daz<2>;\n  .reg .b32 %dbz<2>;\n  .reg .b32 %rs, %ds, %ds0, %k, %kp, %st, %g, %wm, %wn, %ln, %li;")
    e("  .reg .b32 %t, %lane, %w, %M, %N, %KT, %G, %pid, %pidend, %pidm, %pidn, %base, %row, %smem, %rowa, %colb, %npm, %npn;")
    e("  .reg .b64 %pa, %pb, %ps;\n  .reg .b32 %bscale, %colt;")
    e("  ld.param.u32 %M, [pM];\n  ld.param.u32 %N, [pN];\n  ld.param.u32 %G, [pG];")
    e("  ld.param.u64 %d0, [pA];\n  ld.param.u64 %d1, [pB];\n  ld.param.u64 %d2, [pC];")
    e("  ld.param.u64 %d3, [pAs];\n  ld.param.u64 %d4, [pBs];")
    e("  mov.b32 %smem, smem;")
    e("  mov.u32 %t, %tid.x;\n  and.b32 %lane, %t, 31;\n  shr.u32 %w, %t, 5;")
    e("  shr.u32 %wm, %w, 2;\n  and.b32 %wn, %w, 3;")
    e(f"  mov.u32 %KT, {K // BK};")
    e(f"  add.u32 %npm, %M, {BM - 1};\n  shr.u32 %npm, %npm, {LOGM};\n  shr.u32 %npn, %N, 7;")
    e("  mul.lo.u32 %x0, %npm, %npn;")
    e(f"  mov.u32 %pid, %ctaid.x;\n  mul.lo.u32 %pid, %pid, {NB};\n  add.u32 %pidend, %pid, {NB};\n  min.u32 %pidend, %pidend, %x0;")
    e("  and.b32 %ln, %lane, 7;\n  shr.u32 %li, %lane, 3;")
    # A copies: K_SW128 of the row each copy lands on
    e(f"  mul.lo.u32 %x5, %wn, {AF};\n  and.b32 %x5, %x5, 1;\n  shl.b32 %x5, %x5, 2;\n  add.u32 %x5, %x5, %li;")
    e("  xor.b32 %x3, %ln, %x5;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %x4, %x5, 4;\n  and.b32 %x4, %x4, 7;\n  xor.b32 %x4, %ln, %x4;\n  shl.b32 %x4, %x4, 4;")
    e(f"  mul.lo.u32 %rowa, %wn, {AF * 4};\n  add.u32 %rowa, %rowa, %li;\n  shl.b32 %x1, %wm, 6;\n  add.u32 %rowa, %rowa, %x1;")
    e("  shl.b32 %x1, %rowa, 7;\n  add.u32 %x1, %x1, %smem;\n  add.u32 %daz0, %x1, %x3;\n  add.u32 %daz1, %x1, %x4;")
    # B copies
    e("  xor.b32 %x3, %ln, %li;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %x4, %li, 4;\n  xor.b32 %x4, %ln, %x4;\n  shl.b32 %x4, %x4, 4;")
    e("  shl.b32 %colb, %wn, 5;\n  add.u32 %colb, %colb, %li;\n  setp.lt.u32 %pw, %w, 4;")
    e(f"  shl.b32 %x1, %colb, 7;\n  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;\n  add.u32 %dbz0, %x1, %x3;\n  add.u32 %dbz1, %x1, %x4;")
    # scale slots: threads 0..BM-1 fetch a_s of row t, thread BM fetches b_s
    e("  and.b32 %x5, %t, 7;\n  shl.b32 %x5, %x5, 3;\n  shr.u32 %x6, %t, 4;\n  and.b32 %x6, %x6, 3;\n  shl.b32 %x6, %x6, 1;")
    e("  shr.u32 %x7, %t, 3;\n  and.b32 %x7, %x7, 1;\n  add.u32 %x5, %x5, %x6;\n  add.u32 %x5, %x5, %x7;")
    e("  shr.u32 %x6, %t, 6;\n  shl.b32 %x6, %x6, 6;\n  add.u32 %x5, %x5, %x6;\n  shl.b32 %x5, %x5, 2;")
    e(f"  setp.eq.u32 %p1, %t, {BM};\n  setp.lt.u32 %p7, %t, {BM};")
    e(f"  selp.b32 %x5, {max(BM, 64) * 4}, %x5, %p1;\n  add.u32 %x5, %x5, {SS0};\n  add.u32 %ds0, %smem, %x5;")
    # ldmatrix lanes
    e("  shr.u32 %x1, %lane, 3;\n  and.b32 %x1, %x1, 1;\n  shl.b32 %x1, %x1, 3;\n  add.u32 %x1, %x1, %ln;\n  shl.b32 %x1, %x1, 7;")
    e("  shl.b32 %x4, %wm, 13;\n  add.u32 %x1, %x1, %x4;\n  shr.u32 %x2, %lane, 4;")
    for j in range(4):
        e(f"  add.u32 %x3, %x2, {2 * j};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;")
        e(f"  add.u32 %la{j}, %x1, %x3;\n  add.u32 %la{j}, %la{j}, %smem;")
    e("  shl.b32 %x1, %ln, 7;\n  shl.b32 %x4, %wn, 12;\n  add.u32 %x1, %x1, %x4;")
    e(f"  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;")
    for p in range(2):
        e(f"  add.u32 %x3, %li, {4 * p};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %lb{p}, %x1, %x3;")
    e("  shr.u32 %g, %lane, 2;\n  shl.b32 %x1, %g, 5;\n  shl.b32 %x2, %wm, 8;\n  add.u32 %x1, %x1, %x2;")
    e(f"  add.u32 %rs, %x1, {SS0};\n  add.u32 %rs, %rs, %smem;")
    e("  shl.b32 %x2, %wn, 6;\n  and.b32 %x3, %lane, 3;\n  shl.b32 %x3, %x3, 2;\n  add.u32 %colt, %x2, %x3;")

    def copy(pred):
        out = []
        if pred:
            for c in range(AF):
                out.append(f"  and.pred %pq{c}, %pr{c}, {pred};")
        for c in range(AF):
            q = f"@%pq{c}" if pred else f"@%pr{c}"
            out.append(f"  {q} cp.async.cg.shared.global [%da{c & 1} + {c * 512}], [%pa + {4 * c * K}], 16;")
        out.append(f"  and.pred %pz, %pw, {pred};" if pred else "  mov.pred %pz, %pw;")
        for c in range(8):
            out.append(f"  @%pz cp.async.cg.shared.global [%db{c & 1} + {c * 512}], [%pb + {c * 512}], 16;")
        p2 = "%p3" if pred else "%p2"
        out.append(f"  @{p2} ld.global.b32 %x12, [%ps];")
        out.append("  cp.async.commit_group;")
        return out

    def ptr_advance():
        return ["  add.u64 %pa, %pa, 128;", "  add.u64 %pb, %pb, 16384;", "  add.u64 %ps, %ps, 4;"]

    def set_dst(st):
        return [f"  mul.lo.u32 %x8, {st}, {A_STAGE};", "  add.u32 %da0, %daz0, %x8;", "  add.u32 %da1, %daz1, %x8;",
                f"  mul.lo.u32 %x8, {st}, {B_STAGE};", "  add.u32 %db0, %dbz0, %x8;", "  add.u32 %db1, %dbz1, %x8;",
                f"  mul.lo.u32 %x8, {st}, {S_STAGE};", "  add.u32 %ds, %ds0, %x8;"]

    def set_src(st):
        return [f"  mul.lo.u32 %x8, {st}, {A_STAGE};"] + [f"  add.u32 %ra{j}, %la{j}, %x8;" for j in range(4)] + [
                f"  mul.lo.u32 %x8, {st}, {B_STAGE};", "  add.u32 %rb0, %lb0, %x8;", "  add.u32 %rb1, %lb1, %x8;"]

    def ld_a(F, j):
        q = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%ra{j} + {F * 2048}];"

    def ld_b(bset, nb):
        q = ", ".join(f"%B{bset * 16 + nb * 4 + i}" for i in range(4))
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%rb{bset} + {nb * 1024}];"

    def mma(f, F, bset, nb, off, first):
        t = ", ".join(f"%T{f * 4 + i}" for i in range(4))
        a = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        b = f"%B{bset * 16 + nb * 4 + off}, %B{bset * 16 + nb * 4 + off + 1}"
        out = [f"  mov.b32 %T{f * 4 + i}, 0;" for i in range(4)] if first else []
        out.append(f"  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{t}}}, {{{a}}}, {{{b}}}, {{{t}}};")
        return out

    def fma(f):
        F = f >> 2
        return [f"  fma.rn.f32 %C{f * 4 + i}, %T{f * 4 + i}, %S{F * 2 + (i >> 1)}, %C{f * 4 + i};" for i in range(4)]

    e("$BLOCK:")
    e("  setp.ge.u32 %p0, %pid, %pidend;\n  @%p0 bra $END;")
    e("  mul.lo.u32 %x2, %G, %npn;")
    e("  div.u32 %x3, %pid, %x2;\n  mul.lo.u32 %x3, %x3, %G;")
    e("  sub.u32 %x4, %npm, %x3;\n  min.u32 %x4, %x4, %G;")
    e("  rem.u32 %x5, %pid, %x4;\n  add.u32 %pidm, %x3, %x5;")
    e("  rem.u32 %x5, %pid, %x2;\n  div.u32 %pidn, %x5, %x4;")
    e(f"  shl.b32 %base, %pidm, {LOGM};")
    e(f"  add.u32 %row, %base, %rowa;\n  mul.wide.u32 %d5, %row, {K};")
    e("  add.u64 %pa, %d0, %d5;\n  shl.b32 %x1, %ln, 4;\n  cvt.u64.u32 %d6, %x1;\n  add.u64 %pa, %pa, %d6;")
    for c in range(AF):
        e(f"  add.u32 %x1, %row, {4 * c};\n  setp.lt.u32 %pr{c}, %x1, %M;")
    # tile first: (block, tile) is 16 KiB, a row inside it is one 128 B line
    e(f"  mul.lo.u32 %x1, %pidn, {K // BK};\n  mul.wide.u32 %d5, %x1, 16384;\n  add.u64 %pb, %d1, %d5;")
    e("  shl.b32 %x1, %colb, 7;\n  cvt.u64.u32 %d5, %x1;\n  add.u64 %pb, %pb, %d5;\n  add.u64 %pb, %pb, %d6;")
    e("  add.u32 %row, %base, %t;\n  setp.lt.u32 %p2, %row, %M;\n  and.pred %p2, %p2, %p7;\n  or.pred %p2, %p2, %p1;")
    e("  shl.b32 %x4, %KT, 2;\n  mul.wide.u32 %d5, %row, %x4;\n  add.u64 %ps, %d3, %d5;")
    e("  mul.wide.u32 %d6, %pidn, %x4;\n  add.u64 %d6, %d4, %d6;\n  selp.b64 %ps, %d6, %ps, %p1;")
    for i in range(NF * 4):
        e(f"  mov.b32 %C{i}, 0;")
    for f in range(NF - gap, NF):
        for i in range(4):
            e(f"  mov.b32 %T{f * 4 + i}, 0;")
    for i in range(AF * 2):
        e(f"  mov.b32 %S{i}, 0;")
    e("  mov.u32 %k, 0;")
    for i in range(S - 1):
        e(f"  mov.u32 %st, {i};")
        L.extend(set_dst("%st"))
        L.extend(copy(None))
        e("  @%p2 st.shared.b32 [%ds], %x12;")
        L.extend(ptr_advance())
    e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
    e("  mov.u32 %st, 0;")
    L.extend(set_src("%st"))
    for F in range(AF):
        e(ld_a(F, 0))
    for nb in range(4):
        e(ld_b(0, nb))

    e("$LOOP:")
    e(f"  add.u32 %kp, %k, {S - 1};\n  setp.lt.u32 %p4, %kp, %KT;\n  and.pred %p3, %p4, %p2;")
    e(f"  rem.u32 %st, %kp, {S};")
    L.extend(set_dst("%st"))
    cp = copy("%p4") + ptr_advance()

    for j in range(4):
        bset, off = j >> 1, (j & 1) * 2
        idx = 0
        if j == 3:
            e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
            e(f"  add.u32 %kp, %k, 1;\n  rem.u32 %st, %kp, {S};")
            L.extend(set_src("%st"))
        if j == 1:
            e(f"  rem.u32 %x8, %k, {S};\n  mul.lo.u32 %x8, %x8, {S_STAGE};")
            e(f"  add.u32 %x9, %rs, %x8;\n  add.u32 %x10, %smem, {SS0 + max(BM, 64) * 4};\n  add.u32 %x10, %x10, %x8;")
        for F in range(AF):
            for nb in range(4):
                f = F * 4 + nb
                L.extend(mma(f, F, bset, nb, off, j == 0))
                if j == 0:
                    if idx < gap:
                        L.extend(fma(NF - gap + idx))
                    if idx >= gap:
                        take = cp[:cps]
                        del cp[:cps]
                        L.extend(take)
                    if idx == NF - 1:
                        L.extend(cp)
                        cp = []
                elif j == 1:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(1, idx - bpos))
                    if idx == min(4, NF // 4):
                        e("  ld.shared.v4.b32 {%S0, %S1, %S2, %S3}, [%x9];")
                        if AF > 2:
                            e("  ld.shared.v4.b32 {%S4, %S5, %S6, %S7}, [%x9 + 16];")
                        e("  ld.shared.b32 %bscale, [%x10];")
                    if idx >= NF // 2 and idx - NF // 2 < AF * 2 and NF >= 8:
                        e(f"  mul.rn.f32 %S{idx - NF // 2}, %S{idx - NF // 2}, %bscale;")
                elif j == 2:
                    if idx == 0:
                        e("  @%p3 st.shared.b32 [%ds], %x12;")
                elif j == 3:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(0, idx - bpos))
                    if idx >= gap:
                        L.extend(fma(idx - gap))
                idx += 1
            e(ld_a(F, (j + 1) & 3))
        if j == 1 and NF < 8:
            for i in range(AF * 2):
                e(f"  mul.rn.f32 %S{i}, %S{i}, %bscale;")
    e("  add.u32 %k, %k, 1;\n  setp.lt.u32 %p5, %k, %KT;\n  @%p5 bra $LOOP;")
    for f in range(NF - gap, NF):
        L.extend(fma(f))

    e("  shl.b32 %x0, %N, 1;\n  shl.b32 %x1, %pidn, 8;\n  add.u32 %x1, %x1, %colt;\n  cvt.u64.u32 %d7, %x1;")
    e("  add.u64 %d7, %d2, %d7;\n  shl.b32 %x3, %wm, 6;\n  add.u32 %x3, %x3, %base;")
    for F in range(AF):
        for h in range(2):
            e(f"  add.u32 %row, %x3, {16 * F + 8 * h};\n  add.u32 %row, %row, %g;\n  setp.lt.u32 %p6, %row, %M;")
            e("  mul.wide.u32 %d8, %row, %x0;\n  add.u64 %d8, %d7, %d8;")
            for nb in range(4):
                f = F * 4 + nb
                e(f"  cvt.rn.bf16x2.f32 %x11, %C{f * 4 + 2 * h + 1}, %C{f * 4 + 2 * h};")
                e(f"  @%p6 st.global.b32 [%d8 + {16 * nb}], %x11;")
    e("  cp.async.wait_group 0;\n  bar.sync 0;")
    e("  add.u32 %pid, %pid, 1;\n  bra.uni $BLOCK;")
    e("$END:\n  ret;\n}")
    return "\n".join(L) + "\n"


# gate_up GEMM + SiLU * up + down_proj input quant in one kernel. Each warp owns 4 A x 4 B fragments,
# 2 gate + 2 up, so g and u land in the same lane and the activation runs on the accumulators. The
# quant group (128) equals the tile width, so a row's scale is known at block end: absmax is reduced
# through SMEM in three steps on the A buffers, idle by then. One MLP drops from 1.86 GB of activation
# traffic through HBM to 0.14 GB.
FUSED_BM = 64
FUSED_A_STAGE, FUSED_B_STAGE, FUSED_S_STAGE = 0x2000, 0x8000, 0x200
FUSED_LOG2E = "0fBFB8AA3B"      # -log2(e)
FUSED_ONE = "0f3F800000"
FUSED_FP8_MAX = "0f43E00000"    # 448
FUSED_EPS = "0f2EDBE6FF"        # 1e-10
FUSED_NT = 256
FUSED_CONFIG = dict(stages=2, gap=5, NB=2, G=128, cps=8, bpos=3, opt=1)


def fused_smem_bytes(stages):
    return stages * (FUSED_A_STAGE + FUSED_B_STAGE + FUSED_S_STAGE)


def gen_fused(K, stages=2, gap=5, NB=2, cps=8, bpos=3, name="fp8_gemm_silu_quant"):
    S = stages
    SB0 = S * FUSED_A_STAGE
    SS0 = SB0 + S * FUSED_B_STAGE
    R1, R2, R3 = 0, FUSED_A_STAGE, FUSED_A_STAGE + 0x800   # scratch inside the A buffers
    L = []
    e = L.append
    e(".version 8.5\n.target sm_89\n.address_size 64\n.extern .shared .align 16 .b8 smem[];\n")
    e(f".visible .entry {name}(.param .u64 pA, .param .u64 pB, .param .u64 pQ, .param .u64 pAs, .param .u64 pBs, .param .u64 pQs,")
    e("  .param .u32 pM, .param .u32 pN, .param .u32 pG)")
    e(f".reqntid {FUSED_NT}\n{{")
    e("  .reg .b32 %T<64>;\n  .reg .b32 %C<64>;\n  .reg .b32 %A<16>;\n  .reg .b32 %B<32>;\n  .reg .b32 %S<16>;")
    e("  .reg .b32 %x<16>;\n  .reg .b32 %y<4>;\n  .reg .b32 %I<8>;\n  .reg .b16 %h0;\n  .reg .b64 %d<10>;\n  .reg .pred %p<8>;\n  .reg .pred %pr<2>;\n  .reg .pred %pq<2>;\n  .reg .pred %pbu;")
    e("  .reg .b32 %la<4>;\n  .reg .b32 %lb<2>;\n  .reg .b32 %ra<4>;\n  .reg .b32 %rb<2>;\n  .reg .b32 %da<2>;\n  .reg .b32 %db<2>;")
    e("  .reg .b32 %daz<2>;\n  .reg .b32 %dbz<2>;\n  .reg .b32 %rs, %ds, %ds0, %k, %kp, %st, %g, %ln, %li, %t4, %slot;")
    e("  .reg .b32 %t, %lane, %w, %M, %N, %N2, %KT, %G, %pid, %pidend, %pidm, %pidn, %base, %row, %smem, %rowa, %colb, %npm, %npn, %colt;")
    e("  .reg .b64 %pa, %pb, %pu, %ps;\n  .reg .b32 %bg, %bu;")
    e("  ld.param.u32 %M, [pM];\n  ld.param.u32 %N, [pN];\n  ld.param.u32 %G, [pG];\n  shr.u32 %N2, %N, 1;")
    e("  ld.param.u64 %d0, [pA];\n  ld.param.u64 %d1, [pB];\n  ld.param.u64 %d2, [pQ];")
    e("  ld.param.u64 %d3, [pAs];\n  ld.param.u64 %d4, [pBs];\n  ld.param.u64 %d9, [pQs];")
    e("  mov.b32 %smem, smem;")
    e("  mov.u32 %t, %tid.x;\n  and.b32 %lane, %t, 31;\n  shr.u32 %w, %t, 5;\n  and.b32 %t4, %lane, 3;")
    e(f"  mov.u32 %KT, {K // 128};")
    e(f"  add.u32 %npm, %M, {FUSED_BM - 1};\n  shr.u32 %npm, %npm, 6;\n  shr.u32 %npn, %N, 8;")     # gate blocks
    e("  mul.lo.u32 %x0, %npm, %npn;")
    e(f"  mov.u32 %pid, %ctaid.x;\n  mul.lo.u32 %pid, %pid, {NB};\n  add.u32 %pidend, %pid, {NB};\n  min.u32 %pidend, %pidend, %x0;")
    e("  and.b32 %ln, %lane, 7;\n  shr.u32 %li, %lane, 3;")
    # K_SW128 write side
    e("  xor.b32 %x3, %ln, %li;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %x4, %li, 4;\n  xor.b32 %x4, %ln, %x4;\n  shl.b32 %x4, %x4, 4;")
    e("  shl.b32 %rowa, %w, 3;\n  add.u32 %rowa, %rowa, %li;")
    e("  shl.b32 %x1, %rowa, 7;\n  add.u32 %x1, %x1, %smem;\n  add.u32 %daz0, %x1, %x3;\n  add.u32 %daz1, %x1, %x4;")
    e("  shl.b32 %colb, %w, 4;\n  add.u32 %colb, %colb, %li;")
    e(f"  shl.b32 %x1, %colb, 7;\n  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;\n  add.u32 %dbz0, %x1, %x3;\n  add.u32 %dbz1, %x1, %x4;")
    # scale slots: threads 0..63 fetch a_s of row t, 64 and 65 fetch b_s of gate and up
    e("  and.b32 %x5, %t, 7;\n  shl.b32 %x5, %x5, 3;\n  shr.u32 %x6, %t, 4;\n  and.b32 %x6, %x6, 3;\n  shl.b32 %x6, %x6, 1;")
    e("  shr.u32 %x7, %t, 3;\n  and.b32 %x7, %x7, 1;\n  add.u32 %x5, %x5, %x6;\n  add.u32 %x5, %x5, %x7;\n  shl.b32 %slot, %x5, 2;")
    e(f"  setp.eq.u32 %p1, %t, {FUSED_BM};\n  setp.lt.u32 %p7, %t, {FUSED_BM};\n  setp.eq.u32 %pbu, %t, {FUSED_BM + 1};")
    e(f"  selp.b32 %x5, {FUSED_BM * 4}, %slot, %p1;\n  selp.b32 %x5, {FUSED_BM * 4 + 4}, %x5, %pbu;\n  add.u32 %x5, %x5, {SS0};\n  add.u32 %ds0, %smem, %x5;")
    # ldmatrix lanes
    e("  shr.u32 %x1, %lane, 3;\n  and.b32 %x1, %x1, 1;\n  shl.b32 %x1, %x1, 3;\n  add.u32 %x1, %x1, %ln;\n  shl.b32 %x1, %x1, 7;")
    e("  shr.u32 %x2, %lane, 4;")
    for j in range(4):
        e(f"  add.u32 %x3, %x2, {2 * j};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;")
        e(f"  add.u32 %la{j}, %x1, %x3;\n  add.u32 %la{j}, %la{j}, %smem;")
    e("  shl.b32 %x1, %ln, 7;\n  shl.b32 %x4, %w, 11;\n  add.u32 %x1, %x1, %x4;")
    e(f"  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;")
    for p in range(2):
        e(f"  add.u32 %x3, %li, {4 * p};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %lb{p}, %x1, %x3;")
    e("  shr.u32 %g, %lane, 2;\n  shl.b32 %x1, %g, 5;")
    e(f"  add.u32 %rs, %x1, {SS0};\n  add.u32 %rs, %rs, %smem;")
    e("  shl.b32 %x2, %w, 4;\n  shl.b32 %x3, %t4, 1;\n  add.u32 %colt, %x2, %x3;")

    def copy(pred):
        out = []
        if pred:
            for c in range(2):
                out.append(f"  and.pred %pq{c}, %pr{c}, {pred};")
        for c in range(2):
            q = f"@%pq{c}" if pred else f"@%pr{c}"
            out.append(f"  {q} cp.async.cg.shared.global [%da{c & 1} + {c * 512}], [%pa + {4 * c * K}], 16;")
        q = f"@{pred} " if pred else ""
        for c in range(4):
            out.append(f"  {q}cp.async.cg.shared.global [%db{c & 1} + {c * 512}], [%pb + {c * 512}], 16;")
        for c in range(4):
            out.append(f"  {q}cp.async.cg.shared.global [%db{c & 1} + {16384 + c * 512}], [%pu + {c * 512}], 16;")
        p2 = "%p3" if pred else "%p2"
        out.append(f"  @{p2} ld.global.b32 %x12, [%ps];")
        out.append("  cp.async.commit_group;")
        return out

    def ptr_advance():
        return ["  add.u64 %pa, %pa, 128;", "  add.u64 %pb, %pb, 16384;", "  add.u64 %pu, %pu, 16384;", "  add.u64 %ps, %ps, 4;"]

    def set_dst(st):
        return [f"  mul.lo.u32 %x8, {st}, {FUSED_A_STAGE};", "  add.u32 %da0, %daz0, %x8;", "  add.u32 %da1, %daz1, %x8;",
                f"  mul.lo.u32 %x8, {st}, {FUSED_B_STAGE};", "  add.u32 %db0, %dbz0, %x8;", "  add.u32 %db1, %dbz1, %x8;",
                f"  mul.lo.u32 %x8, {st}, {FUSED_S_STAGE};", "  add.u32 %ds, %ds0, %x8;"]

    def set_src(st):
        return [f"  mul.lo.u32 %x8, {st}, {FUSED_A_STAGE};"] + [f"  add.u32 %ra{j}, %la{j}, %x8;" for j in range(4)] + [
                f"  mul.lo.u32 %x8, {st}, {FUSED_B_STAGE};", "  add.u32 %rb0, %lb0, %x8;", "  add.u32 %rb1, %lb1, %x8;"]

    def ld_a(F, j):
        q = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%ra{j} + {F * 2048}];"

    def ld_b(bset, nb):
        q = ", ".join(f"%B{bset * 16 + nb * 4 + i}" for i in range(4))
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%rb{bset} + {(nb & 1) * 1024 + (nb >> 1) * 16384}];"

    def mma(f, F, bset, nb, off, first):
        t = ", ".join(f"%T{f * 4 + i}" for i in range(4))
        a = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        b = f"%B{bset * 16 + nb * 4 + off}, %B{bset * 16 + nb * 4 + off + 1}"
        out = [f"  mov.b32 %T{f * 4 + i}, 0;" for i in range(4)] if first else []
        out.append(f"  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{t}}}, {{{a}}}, {{{b}}}, {{{t}}};")
        return out

    def fma(f):
        F, nb = f >> 2, f & 3
        off = 8 if nb >= 2 else 0
        return [f"  fma.rn.f32 %C{f * 4 + i}, %T{f * 4 + i}, %S{off + F * 2 + (i >> 1)}, %C{f * 4 + i};" for i in range(4)]

    e("$BLOCK:")
    e("  setp.ge.u32 %p0, %pid, %pidend;\n  @%p0 bra $END;")
    e("  mul.lo.u32 %x2, %G, %npn;")
    e("  div.u32 %x3, %pid, %x2;\n  mul.lo.u32 %x3, %x3, %G;")
    e("  sub.u32 %x4, %npm, %x3;\n  min.u32 %x4, %x4, %G;")
    e("  rem.u32 %x5, %pid, %x4;\n  add.u32 %pidm, %x3, %x5;")
    e("  rem.u32 %x5, %pid, %x2;\n  div.u32 %pidn, %x5, %x4;")
    e("  shl.b32 %base, %pidm, 6;")
    e(f"  add.u32 %row, %base, %rowa;\n  mul.wide.u32 %d5, %row, {K};")
    e("  add.u64 %pa, %d0, %d5;\n  shl.b32 %x1, %ln, 4;\n  cvt.u64.u32 %d6, %x1;\n  add.u64 %pa, %pa, %d6;")
    for c in range(2):
        e(f"  add.u32 %x1, %row, {4 * c};\n  setp.lt.u32 %pr{c}, %x1, %M;")
    e(f"  mul.lo.u32 %x1, %pidn, {K // 128};\n  mul.wide.u32 %d5, %x1, 16384;\n  add.u64 %pb, %d1, %d5;")
    e("  shl.b32 %x1, %colb, 7;\n  cvt.u64.u32 %d5, %x1;\n  add.u64 %pb, %pb, %d5;\n  add.u64 %pb, %pb, %d6;")
    e(f"  mul.lo.u32 %x1, %npn, {K // 128};\n  mul.wide.u32 %d5, %x1, 16384;\n  add.u64 %pu, %pb, %d5;")
    e("  add.u32 %row, %base, %t;\n  setp.lt.u32 %p2, %row, %M;\n  and.pred %p2, %p2, %p7;\n  or.pred %p2, %p2, %p1;\n  or.pred %p2, %p2, %pbu;")
    e("  shl.b32 %x4, %KT, 2;\n  mul.wide.u32 %d5, %row, %x4;\n  add.u64 %ps, %d3, %d5;")
    e("  mul.wide.u32 %d6, %pidn, %x4;\n  add.u64 %d6, %d4, %d6;\n  selp.b64 %ps, %d6, %ps, %p1;")
    e("  add.u32 %x1, %pidn, %npn;\n  mul.wide.u32 %d6, %x1, %x4;\n  add.u64 %d6, %d4, %d6;\n  selp.b64 %ps, %d6, %ps, %pbu;")
    for i in range(64):
        e(f"  mov.b32 %C{i}, 0;")
    for f in range(16 - gap, 16):
        for i in range(4):
            e(f"  mov.b32 %T{f * 4 + i}, 0;")
    for i in range(16):
        e(f"  mov.b32 %S{i}, 0;")
    e("  mov.u32 %k, 0;")
    for i in range(S - 1):
        e(f"  mov.u32 %st, {i};")
        L.extend(set_dst("%st"))
        L.extend(copy(None))
        e("  @%p2 st.shared.b32 [%ds], %x12;")
        L.extend(ptr_advance())
    e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
    e("  mov.u32 %st, 0;")
    L.extend(set_src("%st"))
    for F in range(4):
        e(ld_a(F, 0))
    for nb in range(4):
        e(ld_b(0, nb))

    e("$LOOP:")
    e(f"  add.u32 %kp, %k, {S - 1};\n  setp.lt.u32 %p4, %kp, %KT;\n  and.pred %p3, %p4, %p2;")
    e(f"  rem.u32 %st, %kp, {S};")
    L.extend(set_dst("%st"))
    cp = copy("%p4") + ptr_advance()

    for j in range(4):
        bset, off = j >> 1, (j & 1) * 2
        idx = 0
        if j == 3:
            e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
            e(f"  add.u32 %kp, %k, 1;\n  rem.u32 %st, %kp, {S};")
            L.extend(set_src("%st"))
        for F in range(4):
            for nb in range(4):
                f = F * 4 + nb
                L.extend(mma(f, F, bset, nb, off, j == 0))
                if j == 0:
                    if idx < gap:
                        L.extend(fma(16 - gap + idx))
                    if idx >= gap:
                        take = cp[:cps]
                        del cp[:cps]
                        L.extend(take)
                    if idx == 15:
                        L.extend(cp)
                        cp = []
                elif j == 1:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(1, idx - bpos))
                    if idx == 4:
                        e("  ld.shared.v4.b32 {%S0, %S1, %S2, %S3}, [%x9];")
                        e("  ld.shared.v4.b32 {%S4, %S5, %S6, %S7}, [%x9 + 16];")
                        e("  ld.shared.v2.b32 {%bg, %bu}, [%x10];")
                    if idx >= 8:
                        i = idx - 8
                        e(f"  mul.rn.f32 %S{8 + i}, %S{i}, %bu;")
                        e(f"  mul.rn.f32 %S{i}, %S{i}, %bg;")
                elif j == 2:
                    if idx == 0:
                        e("  @%p3 st.shared.b32 [%ds], %x12;")
                elif j == 3:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(0, idx - bpos))
                    if idx >= gap:
                        L.extend(fma(idx - gap))
                idx += 1
            e(ld_a(F, (j + 1) & 3))
        if j == 0:
            e(f"  rem.u32 %x8, %k, {S};\n  mul.lo.u32 %x8, %x8, {FUSED_S_STAGE};")
            e(f"  add.u32 %x9, %rs, %x8;\n  add.u32 %x10, %smem, {SS0 + FUSED_BM * 4};\n  add.u32 %x10, %x10, %x8;")
    e("  add.u32 %k, %k, 1;\n  setp.lt.u32 %p5, %k, %KT;\n  @%p5 bra $LOOP;")
    for f in range(16 - gap, 16):
        L.extend(fma(f))

    # silu(g) * up on the accumulators; the result stays in the gate registers
    for F in range(4):
        for nb in range(2):
            for i in range(4):
                g = f"%C{(F * 4 + nb) * 4 + i}"
                u = f"%C{(F * 4 + nb + 2) * 4 + i}"
                e(f"  mul.rn.f32 %y0, {g}, {FUSED_LOG2E};\n  ex2.approx.ftz.f32 %y0, %y0;\n  add.rn.f32 %y0, %y0, {FUSED_ONE};")
                e(f"  rcp.approx.ftz.f32 %y0, %y0;\n  mul.rn.f32 {g}, {g}, %y0;\n  mul.rn.f32 {g}, {g}, {u};")
    # absmax of each row of this lane, written next to the 32 lanes that share the row
    e(f"  shl.b32 %x1, %g, 7;\n  shl.b32 %x2, %w, 2;\n  add.u32 %x2, %x2, %t4;\n  shl.b32 %x2, %x2, 2;\n  add.u32 %x1, %x1, %x2;\n  add.u32 %x1, %x1, %smem;")
    for F in range(4):
        for h in range(2):
            vals = [f"%C{(F * 4 + nb) * 4 + 2 * h + q}" for nb in range(2) for q in range(2)]
            e(f"  abs.f32 %y1, {vals[0]};")
            for v in vals[1:]:
                e(f"  abs.f32 %y2, {v};\n  max.f32 %y1, %y1, %y2;")
            e(f"  st.shared.b32 [%x1 + {(16 * F + 8 * h) * 128 + R1}], %y1;")
    e("  bar.sync 0;")
    e("  shr.u32 %x1, %t, 2;\n  shl.b32 %x2, %x1, 7;\n  and.b32 %x3, %t, 3;\n  shl.b32 %x4, %x3, 5;\n  add.u32 %x2, %x2, %x4;\n  add.u32 %x2, %x2, %smem;")
    e("  ld.shared.v4.b32 {%I0, %I1, %I2, %I3}, [%x2];\n  ld.shared.v4.b32 {%I4, %I5, %I6, %I7}, [%x2 + 16];")
    for i in range(1, 8):
        e(f"  max.f32 %I0, %I0, %I{i};")
    e(f"  shl.b32 %x2, %x1, 4;\n  shl.b32 %x4, %x3, 2;\n  add.u32 %x2, %x2, %x4;\n  add.u32 %x2, %x2, %smem;\n  st.shared.b32 [%x2 + {R2}], %I0;")
    e("  bar.sync 0;")
    # threads 0..63: one row each
    e(f"  shl.b32 %x2, %t, 4;\n  add.u32 %x2, %x2, %smem;\n  add.u32 %x2, %x2, {R2};")
    e("  @%p7 ld.shared.v4.b32 {%I0, %I1, %I2, %I3}, [%x2];")
    for i in range(1, 4):
        e(f"  max.f32 %I0, %I0, %I{i};")
    e(f"  max.f32 %I0, %I0, {FUSED_EPS};\n  div.rn.f32 %y1, %I0, {FUSED_FP8_MAX};\n  div.rn.f32 %y2, {FUSED_ONE}, %y1;")
    e(f"  add.u32 %x2, %smem, {R3};\n  add.u32 %x2, %x2, %slot;\n  @%p7 st.shared.b32 [%x2], %y2;")
    e("  add.u32 %row, %base, %t;\n  setp.lt.u32 %p6, %row, %M;\n  and.pred %p6, %p6, %p7;")
    e("  shr.u32 %x3, %N, 8;\n  mul.wide.u32 %d5, %row, %x3;\n  cvt.u64.u32 %d6, %pidn;\n  add.u64 %d5, %d5, %d6;\n  shl.b64 %d5, %d5, 2;\n  add.u64 %d5, %d9, %d5;")
    e("  @%p6 st.global.b32 [%d5], %y1;")
    e("  bar.sync 0;")
    # every lane: inverse scales of its 8 rows, then the FP8 store
    e(f"  shl.b32 %x2, %g, 5;\n  add.u32 %x2, %x2, %smem;\n  add.u32 %x2, %x2, {R3};")
    e("  ld.shared.v4.b32 {%I0, %I1, %I2, %I3}, [%x2];\n  ld.shared.v4.b32 {%I4, %I5, %I6, %I7}, [%x2 + 16];")
    e("  shl.b32 %x1, %pidn, 7;\n  add.u32 %x1, %x1, %colt;\n  cvt.u64.u32 %d7, %x1;\n  add.u64 %d7, %d2, %d7;")
    for F in range(4):
        for h in range(2):
            e(f"  add.u32 %row, %base, {16 * F + 8 * h};\n  add.u32 %row, %row, %g;\n  setp.lt.u32 %p6, %row, %M;")
            e("  mul.wide.u32 %d8, %row, %N2;\n  add.u64 %d8, %d7, %d8;")
            for nb in range(2):
                f = F * 4 + nb
                e(f"  mul.rn.f32 %y1, %C{f * 4 + 2 * h}, %I{F * 2 + h};\n  mul.rn.f32 %y2, %C{f * 4 + 2 * h + 1}, %I{F * 2 + h};")
                e("  cvt.rn.satfinite.e4m3x2.f32 %h0, %y2, %y1;")
                e(f"  @%p6 st.global.b16 [%d8 + {8 * nb}], %h0;")
    e("  cp.async.wait_group 0;\n  bar.sync 0;")
    e("  add.u32 %pid, %pid, 1;\n  bra.uni $BLOCK;")
    e("$END:\n  ret;\n}")
    return "\n".join(L) + "\n"


def _cubin(K, c=None):
    tag = "fused" if c is None else "_".join(str(c[k]) for k in ("BM", "stages", "NB", "bpos"))
    return Path(__file__).with_name(f"fp8_hand_sm89_K{K}_{tag}.cubin")


def _ptxas():
    bundled = Path(torch.__file__).parents[1] / "nvidia/cu13/bin/ptxas"
    return str(bundled if bundled.exists() else "ptxas")


def precompile():
    """Run at patch time; the request path never compiles."""
    fc = FUSED_CONFIG
    jobs = [(gen(K, c["stages"], GAP, c["NB"], CPS, c["bpos"], BM=c["BM"]), OPT, _cubin(K, c)) for K in KS for c in _configs(K)]
    jobs.append((gen_fused(FUSED_K, fc["stages"], fc["gap"], fc["NB"], fc["cps"], fc["bpos"]), fc["opt"], _cubin(FUSED_K)))

    def build(job):
        src, opt, out = job
        with tempfile.TemporaryDirectory() as d:
            ptx = f"{d}/k.ptx"
            Path(ptx).write_text(src)
            subprocess.run([_ptxas(), "-arch", "sm_89", "-m64", f"-O{opt}", ptx, "-o", str(out)], check=True)

    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        list(pool.map(build, jobs))
    return [str(job[2]) for job in jobs]


class _Runtime:
    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.fns = {}

    def function(self, K, c=None):
        path = _cubin(K, c)
        fn = self.fns.get(path)
        if fn is None:
            image = ctypes.create_string_buffer(path.read_bytes())
            module = ctypes.c_void_p()
            fn = ctypes.c_void_p()
            name = b"fp8_gemm_silu_quant" if c is None else b"fp8_gemm_hand"
            if self.cu.cuModuleLoadData(ctypes.byref(module), image) or self.cu.cuModuleGetFunction(
                ctypes.byref(fn), module, name
            ):
                raise RuntimeError(f"cannot load {path}")
            self.cu.cuFuncSetCacheConfig(fn, CACHE_ATTR_PREFER_SHARED)
            smem = fused_smem_bytes(FUSED_CONFIG["stages"]) if c is None else smem_bytes(c["stages"], c["BM"])
            if self.cu.cuFuncSetAttribute(fn, MAX_DYNAMIC_SHARED_ATTR, ctypes.c_int(smem)):
                raise RuntimeError(f"cannot set shared memory for {path}")
            self.fns[path] = fn
        return fn


_runtime = None


def _launch(fn, grid, block, smem, values):
    params = (ctypes.c_void_p * len(values))(
        *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values]
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = _runtime.cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, stream, params, None)
    if rc:
        raise RuntimeError(f"cuLaunchKernel failed: {rc}")


def relayout(model):
    """Moves every FP8 linear to tile first storage. The parameter keeps its shape on the meta
    device, so nothing can read the old layout by accident."""
    if not ENABLED or torch.cuda.get_device_capability(0) != (8, 9):
        return
    moved = 0
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale_inv", None)
        if weight is None or scale is None or weight.dtype != torch.float8_e4m3fn or "lm_head" in name:
            continue
        N, K = weight.shape
        if K not in KS or N % BK or tuple(scale.shape) != (N // BK, K // BK) or (K != FUSED_K and N >= 10000):
            logger.info("FP8 hand: %s %s stays on Triton", name, tuple(weight.shape))
            continue
        tiles = weight.data.view(torch.uint8).view(N // BK, BK, K // BK, BK).permute(0, 2, 1, 3).contiguous()
        module.weight = Parameter(torch.empty(N, K, dtype=torch.float8_e4m3fn, device="meta"), requires_grad=False)
        module.weight.tiles = tiles
        moved += 1
    torch.cuda.empty_cache()
    logger.info("FP8 hand: %d linears tile first", moved)


def fused_gate_up(proj, x):
    """Returns (q, scale) for down_proj, or None to run the separate kernels."""
    global _runtime
    if not FUSED_ENABLED or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[0] < FUSED_MIN_M:
        return None
    weight = getattr(proj, "weight", None)
    scale = getattr(proj, "weight_scale_inv", None)
    tiles = getattr(weight, "tiles", None)
    if tiles is None or scale.ndim != 2:
        return None
    N, K = weight.shape
    if K != FUSED_K or N % 256 or getattr(proj, "bias", None) is not None:
        return None
    if _runtime is None:
        _runtime = _Runtime()
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    M = x.shape[0]
    A, As = per_token_group_quant_fp8(x, 128)
    Q = x.new_empty((M, N // 2), dtype=torch.float8_e4m3fn)
    Qs = x.new_empty((M, N // 256), dtype=torch.float32)
    blocks = ((M + FUSED_BM - 1) // FUSED_BM) * (N // 256)
    grid = (blocks + FUSED_CONFIG["NB"] - 1) // FUSED_CONFIG["NB"]
    values = [ctypes.c_uint64(t.data_ptr()) for t in (A, tiles, Q, As, scale, Qs)]
    values += [ctypes.c_uint32(v) for v in (M, N, FUSED_CONFIG["G"])]
    _launch(_runtime.function(K), grid, FUSED_NT, fused_smem_bytes(FUSED_CONFIG["stages"]), values)
    return Q, Qs


def matmul(A, B, C, As, Bs, M, N, K, block_size):
    """Launches C = A @ B^T and returns True, or returns False for a weight left on Triton."""
    global _runtime
    tiles = getattr(B, "tiles", None)
    if tiles is None:
        return False
    if _runtime is None:
        _runtime = _Runtime()
    c = select(N, M)
    BM, NB = c["BM"], c["NB"]
    blocks = ((M + BM - 1) // BM) * (N // BK)
    grid = (blocks + NB - 1) // NB
    values = [ctypes.c_uint64(x.data_ptr()) for x in (A, tiles, C, As, Bs)]
    values += [ctypes.c_uint32(v) for v in (M, N, c["G"])]
    _launch(_runtime.function(K, c), grid, threads(BM), smem_bytes(c["stages"], BM), values)
    return True
