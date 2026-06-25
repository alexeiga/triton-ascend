"""FA accuracy test: CVSplit vs Baseline vs PyTorch reference.
Config: B=1, H=1, N=8192, D=64, BLOCK_M=32, BLOCK_N=32 (large-context inner-loop scale)
  Inner loop: N_CTX/BLOCK_N = 8192/32 = 256 iterations (64 after unroll by 4)
  ISOLATED: only ACTIVE_BLOCKS=1 query tile launched -> 1 AI core (its 2 veccores),
  so sim cost stays ~N=512 level while the inner loop runs the full 64 post-unroll
  iters over the 8192-token context. Accuracy is checked on the computed tile only.
"""
import os
from time import sleep
os.environ["TRITON_ASCEND_SOC_VERSION"] = "Ascend950PR_9589"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch_npu
import triton
import triton.language as tl
import math

CORE_NUM = 32

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale: tl.constexpr,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr,
                    offs_n: tl.constexpr,
                    N_CTX: tl.constexpr,
                    O_base, TRIPWIRE_N: tl.constexpr):
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if TRIPWIRE_N >= 0:
            # Layout-neutral, NON-debug tripwire: a normal GM store whose address
            # goes out-of-bounds ONLY on the target iteration (start_n==TRIPWIRE_N)
            # -> the sim aborts with a recognizable GM address fault at that exact
            # inner-loop step. Off-target iterations write in-bounds (harmless), so
            # no debug mode and no extra cube buffer is introduced.
            _oob = tl.where(start_n == TRIPWIRE_N, 1 << 28, 0)
            tl.store(O_base + _oob + offs_m, m_i.to(tl.float16))
        k = tl.load(K_block_ptr)
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p_cast = p.to(q.type)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale: tl.constexpr, M, Out,
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              TRIPWIRE_N: tl.constexpr):
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H

        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(Q_block_ptr)

        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            task_m_idx, sm_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, STAGE,
            offs_m, offs_n, N_CTX, Out + q_offset, TRIPWIRE_N)

        acc = acc / l_i[:, None]
        m_i += tl.math.log(l_i)
        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def run_attention(q, k, v, sm_scale, block_m, block_n, core_num, active_blocks,
                  tripwire_iter, use_cvsplit=False):
    o = torch.empty_like(q)
    NUM_BLOCKS_M = triton.cdiv(q.shape[2], block_m)
    # Cap the number of query tiles actually launched (isolate the inner loop).
    NUM_BLOCKS = min(NUM_BLOCKS_M * q.shape[0] * q.shape[1], active_blocks)
    grid = (core_num,)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                    device=q.device, dtype=torch.float32)

    kwargs = dict(
        enable_cv_split_scheduling=use_cvsplit,
        cv_split_unroll_factor=int(os.environ.get("CV_SPLIT_UNROLL", "4")),
    ) if use_cvsplit else {}

    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1],
        N_CTX=q.shape[2], HEAD_DIM=q.shape[-1],
        BLOCK_M=block_m, BLOCK_N=block_n, STAGE=0,
        NUM_BLOCKS_M=NUM_BLOCKS_M, NUM_BLOCKS=NUM_BLOCKS,
        AICORE_NUM=core_num,
        TRIPWIRE_N=(tripwire_iter * block_n if tripwire_iter >= 0 else -1),
        **kwargs)
    return o


def main():
    # ---- Run selected variant ----
    # DEBUG tripwire (inert unless FA_TRIPWIRE_ITER is set): fire a device-side
    # assert at a chosen inner-loop iteration so the simulator aborts at a known,
    # recognizable point. Used to bisect the L1->FB dmamov_decode_to_fb abort: the
    # assert is a layout-neutral marker (no extra cube buffer), so the FB bug is not
    # perturbed; whichever fires first (our assert vs dmamov) brackets the culprit
    # iteration. FA_TRIPWIRE_ITER=k makes it trap on the k-th BLOCK_N step.
    tripwire_iter = int(os.environ.get("FA_TRIPWIRE_ITER", "-1"))

    torch.manual_seed(42)

    # Config matched to test_fa_reference_scoped.py (the manually-written target kernel):
    #   Z=1, H=1, N_CTX=256, HEAD_DIM=64, BM=BN=32, non-causal, std=0.5 init, seed 42
    b, h, n, d = 1, 1, 8192, 64
    # Isolate the inner loop: launch only this many query tiles (1 -> 1 AI core / 2
    # veccores). Keeps sim cost ~constant while the inner loop scales with N_CTX.
    active_blocks = 1
    sm_scale = 1.0 / math.sqrt(d)

    # Block sizes are env-overridable so we can test our pass at the target's native
    # geometry (BLOCK_M=BLOCK_N=128 -> ROW_SPLIT 64 rows/veccore, [1,64] row slices),
    # where the per-row SIMD vector path is register-aligned. Default stays 32x32.
    block_m = int(os.environ.get("FA_BLOCK_M", "32"))
    block_n = int(os.environ.get("FA_BLOCK_N", "32"))

    q_npu = torch.empty(b, h, n, d, dtype=torch.float16).normal_(mean=0.0, std=0.5).to("npu")
    k_npu = torch.empty(b, h, n, d, dtype=torch.float16).normal_(mean=0.0, std=0.5).to("npu")
    v_npu = torch.empty(b, h, n, d, dtype=torch.float16).normal_(mean=0.0, std=0.5).to("npu")

    num_blocks_m = triton.cdiv(n, block_m)
    num_blocks = min(num_blocks_m * b * h, active_blocks)
    unroll_factor = int(os.environ.get("CV_SPLIT_UNROLL", "4"))

    # Variant selected by env var so baseline and our pass run as SEPARATE
    # simulator invocations (isolated OPPROF + accuracy, one failure can't block the other):
    #   FA_VARIANT=cvsplit  -> run our CVSplit pass    (default)
    #   FA_VARIANT=baseline -> run baseline (no CVSplit)
    variant = os.environ.get("FA_VARIANT", "cvsplit").lower()
    assert variant in ("cvsplit", "baseline"), f"FA_VARIANT must be cvsplit|baseline, got {variant!r}"
    use_cvsplit = (variant == "cvsplit")
    name = "CVSPLIT (our pass)" if use_cvsplit else "BASELINE (no CVSplit)"

    print("\n" + "="*60)
    print(f"Config: B={b}, H={h}, N={n}, D={d}, BLOCK_M={block_m}, BLOCK_N={block_n}")
    print(f"  Inner loop iters (pre-unroll): {n // block_n}")
    print(f"  Inner loop iters (post-unroll by {unroll_factor}): {n // block_n // unroll_factor}")
    print(f"  ACTIVE query tiles: {num_blocks} (-> {num_blocks} AI core(s), each 2 veccores)")
    print(f"  NUM_BLOCKS={num_blocks}, CORE_NUM={CORE_NUM}")
    print(f"  Running {name}...  [FA_VARIANT={variant}]")
    print(f"  Running {name}...  [UNROLL_FACTOR={unroll_factor}]")
    sleep(10)
    print("="*60)
    try:
        run_attention(q_npu, k_npu, v_npu, sm_scale, block_m, block_n, CORE_NUM,
                      active_blocks, tripwire_iter, use_cvsplit=use_cvsplit)
    except Exception as e:
        print(f"  {variant.upper()} FAILED: {e}")
        return 1

    print("\n" + "="*60)
    print("\nFORWARD RUN DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
