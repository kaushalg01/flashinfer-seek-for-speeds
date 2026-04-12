import triton
import triton.language as tl
import torch 
import math

@triton.jit
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
    page_size: tl.constexpr, topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_CKV: tl.constexpr,
    BLOCK_D_KPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    #each program processes one token's full sparse attention computation
    #we launch num_tokens programs, each streaming over the topk selected K tokens
    #all BLOCK_H heads are processed in parallel within one program
    #NOTE: since all heads share the same sparse indices, we avoid redundant index loads by grouping heads together
    tok_id = tl.program_id(0)

    #offset vectors for indexing into head and dimension axes
    offs_h = tl.arange(0, BLOCK_H)
    #compressed KV dimension offsets, shape: BLOCK_D_CKV
    offs_d_ckv = tl.arange(0, BLOCK_D_CKV)
    #positional encoding dimension offsets, shape: BLOCK_D_KPE
    offs_d_kpe = tl.arange(0, BLOCK_D_KPE)

    #load Q_NOPE tile for this token across all heads
    #shape: [BLOCK_H, BLOCK_D_CKV] — the non-positional component of query
    q_nope_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + offs_d_ckv[None, :] * stride_qt_d
    q_nope = tl.load(q_nope_ptrs)

    #load Q_PE tile for this token across all heads
    #shape: [BLOCK_H, BLOCK_D_KPE] — the positional encoding component of query
    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = tl.load(q_pe_ptrs)

    #online softmax accumulators, maintained per head
    #m_i tracks the running maximum logit for numerical stability
    m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    #l_i tracks the running sum of exponentiated scores (softmax denominator)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    #acc accumulates the weighted value sum, shape: [BLOCK_H, BLOCK_D_CKV]
    acc = tl.zeros([BLOCK_H, BLOCK_D_CKV], dtype=tl.float32)

    #stream over sparse K tokens in tiles of BLOCK_N
    #each iteration processes BLOCK_N of the topk selected tokens
    for n_start in range(0, topk, BLOCK_N):
        #compute token offsets within the sparse index list for this tile
        offs_n = n_start + tl.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k
        
        #mask to handle the last tile where offs_n may exceed topk
        mask_n = offs_n < topk
        
        #load sparse indices — these are global token positions in the paged KV cache
        indices = tl.load(idx_ptrs, mask=mask_n, other=-1)
        #broadcasting masks for 2D operations: col-wise for K loads, row-wise for score masking
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1
        
        #convert global token index to page_idx and token_offset within that page
        page_idx = indices // page_size
        tok_offset = indices % page_size
        
        #load K positional encoding tile from paged cache
        #shape: [BLOCK_N, BLOCK_D_KPE]
        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = tl.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)

        #load compressed KV tile from paged cache (serves as both K_nope and V)
        #shape: [BLOCK_N, BLOCK_D_CKV]
        k_ckv_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = tl.load(k_ckv_ptrs, mask=valid_mask_col, other=0.0)

        #compute attention scores: split into non-positional and positional components
        #qk_nope = Q_nope @ K_ckv^T, shape: [BLOCK_H, BLOCK_N]
        qk_nope = tl.dot(q_nope, k_ckv.T)
        #qk_pe = Q_pe @ K_pe^T, shape: [BLOCK_H, BLOCK_N]
        qk_pe = tl.dot(q_pe, k_kpe.T)
        
        #combined attention logits scaled by sm_scale (1/sqrt(d))
        qk = (qk_nope + qk_pe) * sm_scale
        
        #mask out invalid positions (padding from last tile)
        qk = tl.where(valid_mask_row, qk, -float("inf"))

        #online softmax: update running max, rescale previous accumulator, and add new contributions
        #this avoids materialising the full attention matrix across all topk tokens
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        
        #rescaling factor for previously accumulated values when max changes
        alpha = tl.exp(m_i - m_ij)
        
        #update softmax denominator with rescaled old sum + new sum
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij
        
        #rescale previous accumulator and add new weighted values
        #NOTE: k_ckv is reused here as V since DeepSeek MLA shares compressed KV
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), k_ckv)

    #finalise output: normalise accumulated values by softmax denominator
    #handle edge case where a head saw no valid tokens (m_i stays -inf)
    m_i_finite = m_i != -float("inf")
    m_i_finite_col = m_i[:, None] != -float("inf")
    
    #divide by l_i to complete the softmax-weighted average; zero out heads with no valid tokens
    acc_out = tl.where(m_i_finite_col, acc / l_i[:, None], 0.0)
    
    #store the attention output, shape: [BLOCK_H, BLOCK_D_CKV]
    out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
    tl.store(out_ptrs, acc_out.to(tl.bfloat16))
    
    #compute log-sum-exp in base-2 for numerical stability export
    #LSE = (m + log(l)) / log(2), used downstream for multi-split merging
    math_log2 = 0.6931471805599453
    lse = (m_i + tl.log(l_i)) / math_log2
    lse_out = tl.where(m_i_finite, lse, -float("inf"))
    
    #store per-head LSE values for this token
    lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse_out)

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
