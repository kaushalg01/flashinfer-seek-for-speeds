import triton
import triton.language as tl
import torch 
import math

# ── Local kernel imports ──────────────────────────────────────────────────────
from .indexer_kernel import run_indexer_and_topk   # TopK indexer + selector

def dsa_fwd_kernel(
    Q_NOPE, Q_PE, CKV_CACHE, KPE_CACHE, SPARSE_INDICES, OUTPUT, LSE,
    sm_scale,
    stride_qt_tok, stride_qt_h, stride_qt_d,
    stride_qpe_tok, stride_qpe_h, stride_qpe_d,
    stride_ckv_page, stride_ckv_tok, stride_ckv_d,
    stride_kpe_page, stride_kpe_tok, stride_kpe_d,
    stride_idx_tok, stride_idx_k,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    page_size: t1.constexpr, topk: t1.constexpr,
    BLOCK_N: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_D_KPE: t1.constexpr,
    BLOCK_H: t1.constexpr
):
    #each program processes one token's full sparse attention computation
    #we launch num_tokens programs, each streaming over the topk selected K tokens
    #all BLOCK_H heads are processed in parallel within one program
    #NOTE: since all heads share the same sparse indices, we avoid redundant index loads by grouping heads together
    tok_id = t1.program_id(0)

    #offset vectors for indexing into head and dimension axes
    offs_h = t1.arange(0, BLOCK_H)
    #compressed KV dimension offsets, shape: BLOCK_D_CKV
    offs_d_ckv = t1.arange(0, BLOCK_D_CKV)
    #positional encoding dimension offsets, shape: BLOCK_D_KPE
    offs_d_kpe = t1.arange(0, BLOCK_D_KPE)

    #load Q_NOPE tile for this token across all heads
    #shape: [BLOCK_H, BLOCK_D_CKV] — the non-positional component of query
    q_nope_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + offs_d_ckv[None, :] * stride_qt_d
    q_nope = t1.load(q_nope_ptrs)

    #load Q_PE tile for this token across all heads
    #shape: [BLOCK_H, BLOCK_D_KPE] — the positional encoding component of query
    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = t1.load(q_pe_ptrs)

    #online softmax accumulators, maintained per head
    #m_i tracks the running maximum logit for numerical stability
    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    #l_i tracks the running sum of exponentiated scores (softmax denominator)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    #acc accumulates the weighted value sum, shape: [BLOCK_H, BLOCK_D_CKV]
    acc = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)

    #stream over sparse K tokens in tiles of BLOCK_N
    #each iteration processes BLOCK_N of the topk selected tokens
    for n_start in range(0, topk, BLOCK_N):
        #compute token offsets within the sparse index list for this tile
        offs_n = n_start + t1.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k
        
        #mask to handle the last tile where offs_n may exceed topk
        mask_n = offs_n < topk
        
        #load sparse indices — these are global token positions in the paged KV cache
        indices = t1.load(idx_ptrs, mask=mask_n, other=-1)
        #broadcasting masks for 2D operations: col-wise for K loads, row-wise for score masking
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1
        
        #convert global token index to page_idx and token_offset within that page
        page_idx = indices // page_size
        tok_offset = indices % page_size
        
        #load K positional encoding tile from paged cache
        #shape: [BLOCK_N, BLOCK_D_KPE]
        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = t1.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)

        #load compressed KV tile from paged cache (serves as both K_nope and V)
        #shape: [BLOCK_N, BLOCK_D_CKV]
        k_ckv_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = t1.load(k_ckv_ptrs, mask=valid_mask_col, other=0.0)

        #compute attention scores: split into non-positional and positional components
        #qk_nope = Q_nope @ K_ckv^T, shape: [BLOCK_H, BLOCK_N]
        qk_nope = t1.dot(q_nope, k_ckv.T)
        #qk_pe = Q_pe @ K_pe^T, shape: [BLOCK_H, BLOCK_N]
        qk_pe = t1.dot(q_pe, k_kpe.T)
        
        #combined attention logits scaled by sm_scale (1/sqrt(d))
        qk = (qk_nope + qk_pe) * sm_scale
        
        #mask out invalid positions (padding from last tile)
        qk = t1.where(valid_mask_row, qk, -float("inf"))

        #online softmax: update running max, rescale previous accumulator, and add new contributions
        #this avoids materialising the full attention matrix across all topk tokens
        m_ij = t1.maximum(m_i, t1.max(qk, axis=1))
        p = t1.exp(qk - m_ij[:, None])
        
        #rescaling factor for previously accumulated values when max changes
        alpha = t1.exp(m_i - m_ij)
        
        #update softmax denominator with rescaled old sum + new sum
        l_i = l_i * alpha + t1.sum(p, axis=1)
        m_i = m_ij
        
        #rescale previous accumulator and add new weighted values
        #NOTE: k_ckv is reused here as V since DeepSeek MLA shares compressed KV
        acc = acc * alpha[:, None]
        acc += t1.dot(p.to(t1.bfloat16), k_ckv)

    #finalise output: normalise accumulated values by softmax denominator
    #handle edge case where a head saw no valid tokens (m_i stays -inf)
    m_i_finite = m_i != -float("inf")
    m_i_finite_col = m_i[:, None] != -float("inf")
    
    #divide by l_i to complete the softmax-weighted average; zero out heads with no valid tokens
    acc_out = t1.where(m_i_finite_col, acc / l_i[:, None], 0.0)
    
    #store the attention output, shape: [BLOCK_H, BLOCK_D_CKV]
    out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
    t1.store(out_ptrs, acc_out.to(t1.bfloat16))
    
    #compute log-sum-exp in base-2 for numerical stability export
    #LSE = (m + log(l)) / log(2), used downstream for multi-split merging
    math_log2 = 0.6931471805599453
    lse = (m_i + t1.log(l_i)) / math_log2
    lse_out = t1.where(m_i_finite, lse, -float("inf"))
    
    #store per-head LSE values for this token
    lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
    t1.store(lse_ptrs, lse_out)

def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    #extract tensor dimensions for kernel configuration
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]
    
    #static assertions matching the DeepSeek MLA configuration
    #these allow Triton to use compile-time constants for optimal register allocation
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    
    device = q_nope.device
    
    #allocate output tensors: attention result and log-sum-exp per head
    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)
    
    #launch one program per token — each program handles all heads for that token
    grid = (num_tokens,)
    
    #BLOCK_N defines how many sparse K tokens are processed per iteration of the inner loop
    BLOCK_N = 64
    
    dsa_fwd_kernel[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        page_size=page_size, topk=topk,
        BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_D_KPE=head_dim_kpe,
        BLOCK_H=num_qo_heads,
    )
    
    return output, lse    

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
    device: str = "cuda"
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
cale : float
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
    output, lse = kernel(
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
