"""
wrapper.py
----------
End-to-end pipeline that chains the two Triton kernels together:

  1. TopK Indexer  (indexer_kernel.py)
     - Computes per-token importance scores for the KV cache using
       FP8 index-head dot-products weighted per head.
     - Selects the top-K token indices for each batch entry.

  2. DeepSeek-Style Sparse Attention  (kernel.py)
     - Runs the forward pass of MLA sparse attention using only the
       top-K tokens selected by the indexer as the key/value set.

Entry point
-----------
    output, lse = run_sparse_attention_pipeline(...)

See the docstring of `run_sparse_attention_pipeline` for the full
parameter description and tensor shapes.
"""

import torch

# ── Local kernel imports ──────────────────────────────────────────────────────
from indexer_kernel import run_indexer_and_topk   # TopK indexer + selector
from kernel import kernel as dsa_kernel            # DSA sparse attention forward


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_sparse_attention_pipeline(
    # ── Indexer inputs ────────────────────────────────────────────────────────
    q_index_fp8: torch.Tensor,        # [batch, num_index_heads, index_head_dim] fp8
    k_index_cache_fp8: torch.Tensor,  # [num_pages * page_size * head_dim_with_scale] fp8
    weights: torch.Tensor,            # [batch, num_index_heads] float32  — per-head importance weights
    seq_lens: torch.Tensor,           # [batch] int32  — actual sequence length per sample
    block_table: torch.Tensor,        # [batch, max_num_pages] int32  — page-table for paged KV cache
    seq_offsets: torch.Tensor,        # [batch] int32  — cumulative sum of seq_lens (token start offsets)
    # ── Indexer config ────────────────────────────────────────────────────────
    num_index_heads: int,
    index_head_dim: int,
    num_pages: int,
    page_size: int,
    kv_cache_num_heads: int,
    head_dim_with_scale: int,         # index_head_dim + 1 scale element per token
    max_num_pages: int,
    topk: int,                        # number of top-K tokens to select per sequence
    # ── Sparse-attention inputs ───────────────────────────────────────────────
    q_nope: torch.Tensor,             # [num_tokens, num_qo_heads, head_dim_ckv] bfloat16
    q_pe: torch.Tensor,               # [num_tokens, num_qo_heads, head_dim_kpe] bfloat16
    ckv_cache: torch.Tensor,          # [num_pages, page_size, head_dim_ckv]  bfloat16
    kpe_cache: torch.Tensor,          # [num_pages, page_size, head_dim_kpe]  bfloat16
    sm_scale: float,                  # softmax scale, typically 1 / sqrt(head_dim)
    # ── Tuning knobs (indexer) ────────────────────────────────────────────────
    BLOCK_TOKENS: int = 32,
    BLOCK_HEADS: int = 8,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full pipeline: TopK indexer → DSA sparse attention.

    Parameters
    ----------
    q_index_fp8 : Tensor [batch, num_index_heads, index_head_dim]
        FP8-quantised query used by the indexer to score KV tokens.
    k_index_cache_fp8 : Tensor [num_pages * page_size * head_dim_with_scale]
        Flat paged KV cache for the indexer (FP8 + per-token scale appended).
    weights : Tensor [batch, num_index_heads]
        Per-head importance weights applied after dot-product scoring.
    seq_lens : Tensor [batch]
        True sequence lengths (in tokens) for each batch entry.
    block_table : Tensor [batch, max_num_pages]
        Page table mapping (batch, page_id) → physical page index.
    seq_offsets : Tensor [batch]
        Cumulative token offsets; seq_offsets[i] = sum(seq_lens[:i]).
    num_index_heads : int
        Number of index heads used for scoring (e.g. 8).
    index_head_dim : int
        Dimension of each index head (e.g. 128).
    num_pages : int
        Total number of physical pages in the KV cache.
    page_size : int
        Tokens per physical page (e.g. 64).
    kv_cache_num_heads : int
        Number of heads in the main KV cache (for stride calculation).
    head_dim_with_scale : int
        index_head_dim + 1 (one trailing scalar scale per token per head).
    max_num_pages : int
        Maximum pages a single sequence can occupy (for block-table stride).
    topk : int
        How many tokens to select per sequence for sparse attention.
    q_nope : Tensor [num_tokens, num_qo_heads, head_dim_ckv]
        Non-positional query component for DSA forward.
    q_pe : Tensor [num_tokens, num_qo_heads, head_dim_kpe]
        Positional-encoding query component for DSA forward.
    ckv_cache : Tensor [num_pages, page_size, head_dim_ckv]
        Compressed KV cache (K_nope == V in DeepSeek MLA).
    kpe_cache : Tensor [num_pages, page_size, head_dim_kpe]
        Positional-encoding K cache.
    sm_scale : float
        Softmax temperature scale, usually 1/sqrt(head_dim_ckv + head_dim_kpe).
    BLOCK_TOKENS : int
        Triton tile width along the token dimension (indexer + topk).
    BLOCK_HEADS : int
        Number of query heads processed in one Triton program (indexer).
    device : str
        Target CUDA device string (passed to the indexer allocator).

    Returns
    -------
    output : Tensor [num_tokens, num_qo_heads, head_dim_ckv]  bfloat16
        Sparse-attention output.
    lse : Tensor [num_tokens, num_qo_heads]  float32
        Log-sum-exp (base-2) per head, useful for multi-split merging.
    """

    batch_size = seq_lens.shape[0]

    # ── Step 1: TopK Indexer ──────────────────────────────────────────────────
    # Scores every KV token with a lightweight FP8 dot-product, then picks the
    # top-K global token indices per batch entry.
    #
    # Returns sparse_indices : [batch_size, topk]  int32
    sparse_indices = run_indexer_and_topk(
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_offsets=seq_offsets,
        batch_size=batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        num_pages=num_pages,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,
        topk=topk,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HEADS=BLOCK_HEADS,
        device=device,
    )
    # sparse_indices shape: [batch_size, topk]
    # Each row lists the global token positions (across the paged cache) that
    # scored highest for that sequence — these become the sparse K/V set.

    # ── Step 2: Broadcast sparse indices to match query token dimension ───────
    # `q_nope` is laid out as [num_tokens, …] where num_tokens == sum(seq_lens).
    # The DSA kernel expects sparse_indices shaped [num_tokens, topk], with each
    # token row containing its batch's selected indices.
    num_tokens = q_nope.shape[0]

    # Expand per-batch indices to per-token indices using seq_lens as a repeat count.
    #   seq_lens[i] tells how many query tokens belong to batch i.
    #   torch.repeat_interleave replicates row i of sparse_indices seq_lens[i] times.
    sparse_indices_per_token = torch.repeat_interleave(
        sparse_indices, seq_lens.to(torch.long), dim=0
    )  # [num_tokens, topk]

    assert sparse_indices_per_token.shape == (num_tokens, topk), (
        f"Shape mismatch after broadcast: expected ({num_tokens}, {topk}), "
        f"got {sparse_indices_per_token.shape}"
    )

    # ── Step 3: DSA Sparse Attention ──────────────────────────────────────────
    # Runs the full DeepSeek-Style Sparse Attention forward pass over the
    # top-K tokens identified by the indexer.
    output, lse = dsa_kernel(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices_per_token,
        sm_scale=sm_scale,
    )
    # output : [num_tokens, num_qo_heads, head_dim_ckv]  bfloat16
    # lse    : [num_tokens, num_qo_heads]                float32

    return output, lse


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (python wrapper.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    device = "cuda"

    # ── Dimensions matching DeepSeek MLA config ───────────────────────────────
    batch_size     = 2
    num_qo_heads   = 16
    head_dim_ckv   = 512
    head_dim_kpe   = 64
    page_size      = 64
    topk           = 2048

    # Indexer config
    num_index_heads   = 8
    index_head_dim    = 128
    head_dim_with_scale = index_head_dim + 1   # fp8 dim + 1 scale
    kv_cache_num_heads  = num_qo_heads
    max_num_pages       = 32
    num_pages           = max_num_pages * batch_size

    # Sequence lengths (varied per batch)
    seq_lens    = torch.tensor([128, 192], dtype=torch.int32, device=device)
    seq_offsets = torch.cat([torch.tensor([0], device=device),
                             seq_lens.cumsum(0)[:-1]]).to(torch.int32)
    num_tokens  = seq_lens.sum().item()

    # ── Allocate dummy tensors ────────────────────────────────────────────────
    q_index_fp8 = torch.randint(
        -128, 127,
        (batch_size, num_index_heads, index_head_dim),
        dtype=torch.int8, device=device,
    )
    k_index_cache_fp8 = torch.randint(
        -128, 127,
        (num_pages * page_size * head_dim_with_scale,),
        dtype=torch.int8, device=device,
    )
    weights     = torch.rand(batch_size, num_index_heads, device=device)
    block_table = torch.randint(0, num_pages,
                                (batch_size, max_num_pages),
                                dtype=torch.int32, device=device)

    q_nope  = torch.randn(num_tokens, num_qo_heads, head_dim_ckv,
                          dtype=torch.bfloat16, device=device)
    q_pe    = torch.randn(num_tokens, num_qo_heads, head_dim_kpe,
                          dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv,
                            dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe,
                            dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print("Running sparse attention pipeline...")
    output, lse = run_sparse_attention_pipeline(
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_offsets=seq_offsets,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        num_pages=num_pages,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,
        topk=topk,
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sm_scale=sm_scale,
    )

    print(f"output shape : {output.shape}  dtype={output.dtype}")
    print(f"lse shape    : {lse.shape}     dtype={lse.dtype}")
    print("Pipeline completed successfully.")
