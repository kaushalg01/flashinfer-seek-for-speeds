import torch
import triton
import triton.language as t1
@triton.jit

def indexer_kernel(
    q_index_fp8,           # [batch_size, num_index_heads, index_head_dim] fp8
    k_index_cache_fp8,     # [num_pages * page_size * head_dim_with_scale] fp8
    weights,               # [batch_size, num_index_heads] float32
    seq_lens,              # [batch_size] int32
    block_table,           # [batch_size, max_num_pages] int32
    seq_offsets,           # [batch_size] int32, cumulative sum of seq_lens
    tile_offsets_ptr,      # cumulative tile offsets (pid → batch mapping)

    acc_ptr,
    batch_size,
    num_index_heads,
    index_head_dim,
    page_size,
    kv_cache_num_heads,
    head_dim_with_scale,
    max_num_pages,

    BLOCK_TOKENS: t1.constexpr,
    BLOCK_HEADS: t1.constexpr
):

    # Each program processes a tile of tokens for a given sequence (batch element),
    # instead of splitting work across heads.

    # We divide the KV cache into pages and further into token tiles.
    # For each program:
    #   - it owns a BLOCK_TOKENS chunk of tokens for one sequence
    #   - it computes the final score for those tokens across ALL heads

    # Execution flow:
    # 1. Map program_id → (batch_id, token_tile_id) using tile_offsets
    # 2. For the given sequence:
    #    - fetch its KV cache pages using block_table
    # 3. Load a tile of K (for the current token tile)
    #    - dequantize FP8 values using per-token scales
    # 4. For each head:
    #    - load corresponding Q vector
    #    - compute dot(Q, K_tile) → scores per token
    #    - apply activation (ReLU) and head-specific weights
    #    - accumulate into token_scores

    # Key design decisions:
    # - Parallelism is across tokens (not heads)
    # - Each token is processed by exact1y one program → Earlier, each program was handling some BLOCK_HEADS and writing result per token in a register
    #   This writing was atomic and was causing serialisation of programs, breaking parallelism
    # - K tiles are loaded once per page and reused across all heads (improves memory efficiency)
    # - Q is reloaded per head (acceptable since Q is small compared to K)

    # Total programs launched:
    #   sum_over_batch ceil(seq_len / BLOCK_TOKENS)

    # This design:
    #   - eliminates contention from atomic adds (against earlier implementation)
    #   - improves effective memory bandwidth usage
    #   - maintains good reuse of K across heads within a program

    # Conceptual layout:
    #
    # For each sequence:
    #   tokens → split across programs
    #   heads  → processed inside each program (reduction dimension)
    #
    #        head_id
    #      0   1   2
    # token
    # 0    h0  h1  h2
    # 1    h3  h4  h5
    # 2    h6  h7  h8
    #
    # Each program handles a vertical slice (tokens),
    # and reduces across all heads locally.

    # -------------------------------------------------------
    # PROGRAM MAPPING
    # program = (batch_id, token_tile)
    # -------------------------------------------------------
    pid = t1.program_id(0)

    # -------------------------------------------------------
    # FIND batch_id USING tile_offsets
    # -------------------------------------------------------
    # -------------------------------------------------------
    # PROGRAM MAPPING: pid → (batch_id, token_tile)
    # -------------------------------------------------------

    batch_id = 0
    for b in range(batch_size):
        if pid < t1.load(tile_offsets_ptr + b):
            batch_id = b
            break

    prev_offset = t1.where(
        batch_id > 0,
        t1.load(tile_offsets_ptr + batch_id - 1),
        0
    )

    token_tile_id = pid - prev_offset

    # -------------------------------------------------------
    # LOAD SEQUENCE METADATA
    # -------------------------------------------------------
    seq_len = t1.load(seq_lens + batch_id)
    seq_start = t1.load(seq_offsets + batch_id)

    # -------------------------------------------------------
    # PAGE-ALIGNED TOKEN TILE
    # -------------------------------------------------------
    token_start = token_tile_id * BLOCK_TOKENS

    # compute page id for this tile
    page_id = token_start // page_size
    offset_in_page = token_start % page_size

    # clamp so tile does not cross page
    tokens_left_in_page = page_size - offset_in_page
    effective_tokens = t1.minimum(BLOCK_TOKENS, tokens_left_in_page)

    offs_t = t1.arange(0, BLOCK_TOKENS)
    offset_token = token_start + offs_t

    token_mask = offset_token < (token_start + effective_tokens)
    token_mask &= offset_token < seq_len

    offs_d = t1.arange(0, index_head_dim)

    # -------------------------------------------------------
    # FETCH PAGE POINTER (ONLY ONE PAGE)
    # -------------------------------------------------------
    page_index = t1.load(
        block_table + batch_id * max_num_pages + page_id
    )

    k_page_ptr = k_index_cache_fp8 + (
        page_index * page_size * head_dim_with_scale * kv_cache_num_heads
    )

    # -------------------------------------------------------
    # LOAD K TILE
    # -------------------------------------------------------
    k_ptrs = (
        k_page_ptr
        + (offset_in_page + offs_t)[:, None] * head_dim_with_scale
        + offs_d[None, :]
    )

    k_tile = t1.load(
        k_ptrs,
        mask=token_mask[:, None],
        other=0.0
    )

    scale_ptrs = (
        k_page_ptr
        + (offset_in_page + offs_t) * head_dim_with_scale
        + index_head_dim
    )

    scale_vals = t1.load(
        scale_ptrs,
        mask=token_mask,
        other=0.0
    )

    # dequantize
    k_vals = k_tile.to(t1.float16) * scale_vals[:, None]

    # -------------------------------------------------------
    # ACCUMULATE ACROSS HEADS (REDUCTION INSIDE PROGRAM)
    # -------------------------------------------------------
    token_scores = t1.zeros([BLOCK_TOKENS], t1.float32)

    for h_block in range(0, num_index_heads, BLOCK_HEADS):

    # ---------------------------------------------
    # LOAD Q BLOCK [BLOCK_HEADS, head_dim]
    # ---------------------------------------------
        offs_h = t1.arange(0, BLOCK_HEADS)
        h_ids = h_block + offs_h

        q_ptrs = (
            q_index_fp8
            + batch_id * num_index_heads * index_head_dim
            + offs_d[:, None]                        # dim is now primary
            + h_ids[None, :] * index_head_dim
        )

        q_block = t1.load(
            q_ptrs,
            mask=(offs_d[:, None] < index_head_dim) & (h_ids[None, :] < num_index_heads),
            other=0.0
        )
        q_block = t1.trans(q_block) # this is just for correcting math
    # ---------------------------------------------
    # LOAD WEIGHTS [BLOCK_HEADS]
    # ---------------------------------------------
        w_ptrs = weights + batch_id * num_index_heads + h_ids

        w_block = t1.load(
            w_ptrs,
            mask=h_ids < num_index_heads,
            other=0.0
        )

    # ---------------------------------------------
    # COMPUTE: [tokens, dim] × [heads, dim]
    # ---------------------------------------------
    # k_vals: [BLOCK_TOKENS, dim]
    # q_block: [BLOCK_HEADS, dim]

    # broadcast multiply → [BLOCK_HEADS, BLOCK_TOKENS]
        # q_block is now [dim, heads]

        scores = t1.sum(
            q_block[None, :, :] * k_vals[:, None, :],
            axis=2
        )

    # activation
        scores = t1.maximum(scores, 0.0)

    # apply weights
        # scores = scores * w_block[:, None]
        scores = scores * w_block[None, :] 
        # please check which one works better and is better

    # reduce across heads
        token_scores += t1.sum(scores, axis=0)

    # global token indices
    global_token_ids = seq_start + offset_token
    
    # store results
    t1.store(
        acc_ptr + global_token_ids,
        token_scores,
        mask=token_mask
    )


@triton.jit
def topk_kernel(
    acc_ptr,
    seq_offsets,
    seq_lens,
    topk_indices_ptr,
    K,

    BLOCK_TOKENS: t1.constexpr,
    MAX_K: t1.constexpr,
):
    batch_id = t1.program_id(0)

    seq_start = t1.load(seq_offsets + batch_id)
    seq_len   = t1.load(seq_lens + batch_id)

    # running topK buffers (registers)
    top_scores  = t1.full([MAX_K], -1e9, t1.float32)
    top_indices = t1.zeros([MAX_K], t1.int32)

    for token_start in range(0, seq_len, BLOCK_TOKENS):

        offs = token_start + t1.arange(0, BLOCK_TOKENS)
        mask = offs < seq_len

        # load from acc_ptr
        scores = t1.load(
            acc_ptr + seq_start + offs,
            mask=mask,
            other=-1e9,
        )

        global_ids = seq_start + offs

        # merge candidates
        merged_scores = t1.zeros([2 * MAX_K], t1.float32)
        merged_indices = t1.zeros([2 * MAX_K], t1.int32)

        merged_scores[:MAX_K] = top_scores
        merged_scores[MAX_K:] = scores

        merged_indices[:MAX_K] = top_indices
        merged_indices[MAX_K:] = global_ids

        # sort
        order = t1.argsort(merged_scores, descending=True)

        top_scores  = merged_scores[order[:K]]
        top_indices = merged_indices[order[:K]]

    # store results
    offs_k = t1.arange(0, K)

    t1.store(
        topk_indices_ptr + batch_id * K + offs_k,
        top_indices,
    )

def run_indexer_and_topk(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
    seq_offsets,
    batch_size,
    num_index_heads,
    index_head_dim,
    page_size,
    kv_cache_num_heads,
    head_dim_with_scale,
    max_num_pages,
    topk,
    BLOCK_TOKENS=32,
    BLOCK_HEADS=8
    device='cuda',
):

    # -------------------------------------------------------
    # STEP 0: compute tile_offsets (pid → batch mapping)
    # -------------------------------------------------------
    tiles_per_seq = (seq_lens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    tile_offsets = torch.cumsum(tiles_per_seq, dim=0)

    total_tiles = tile_offsets[-1].item()

    # -------------------------------------------------------
    # STEP 1: allocate per-token scores (NO atomics)
    # -------------------------------------------------------
    total_tokens = seq_offsets[-1] + seq_lens[-1]

    acc = torch.zeros(total_tokens, device=device, dtype=torch.float32)

    # -------------------------------------------------------
    # STEP 2: launch indexer kernel
    # -------------------------------------------------------
    grid = (total_tiles,)

    indexer_kernel[grid](
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_offsets=seq_offsets,
        tile_offsets_ptr=tile_offsets,
        acc_ptr=acc,

        batch_size=batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,

        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HEADS=BLOCK_HEADS
    )

    # -------------------------------------------------------
    # STEP 3: allocate topk output
    # -------------------------------------------------------
    topk_indices = torch.zeros(
        (batch_size, topk),
        device=device,
        dtype=torch.int32
    )

    # -------------------------------------------------------
    # STEP 4: launch topk kernel
    # -------------------------------------------------------
    topk_grid = (batch_size,)

    topk_kernel[topk_grid](
        acc_ptr=acc,
        seq_offsets=seq_offsets,
        seq_lens=seq_lens,
        topk_indices_ptr=topk_indices,
        K=topk,

        BLOCK_TOKENS=BLOCK_TOKENS,
        MAX_K=topk,   # static upper bound
    )

    return topk_indices

# the below mentioned function provides the golden function against which the kernel is to be compared
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 64
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return output, lse
