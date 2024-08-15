#  vLLM Paged Attention

## paged_attention_v1

* input
  * ``out: shape [num_seqs, num_heads, head_size]``
  * ``query: shape [num_seqs, num_heads, head_size]``
  * ``key_cache: shape [num_blocks, num_heads, head_size/x, block_size, x]``
  * ``value_cache: shape [num_blocks, num_heads, head_size, block_size]``
  * ``block_tables: shape [num_seqs, max_num_blocks_per_seq]``
  * ``num_kv_heads: num_heads``
  * ``context_lens: num_seqs``

x代表的是一个向量化的大小



* CUDA设置
  * ``gird: shape (num_heads, num_seqs, num_partition)`` 其中num_partition在不采用的时候为1
  * ``block: shape (NUM_THREADS)``





其中，对于attn_metadata，prefill的数据在前面，decode的数据在后面

```python
# NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|
    
num_prefills=num_prefills,
slot_mapping=slot_mapping_tensor, # token对应在table中的slot id
num_prefill_tokens=num_prefill_tokens, # prefill token的数目
num_decode_tokens=num_decode_tokens, # decode token的数目
seq_lens=seq_lens, # 各个句子的长度
seq_lens_tensor=seq_lens_tensor, # tensor类型的seq_lens，和上面没什么区别
max_query_len=max_query_len, # prefill阶段的query最大值，假如采用了chunk prefill，query_len，而不是context_len。比如484第一次chunked prefill算了20，则第二次max_query_len为464
max_prefill_seq_len=max_prefill_seq_len, # 可看上图
max_decode_seq_len=max_decode_seq_len, # Maximum sequence length among decode batch. 0 if there are prefill requests only.
query_start_loc=query_start_loc, # if the subquery length is [4, 6], it is [0, 4, 10]. 这个是query，假如decode，query是1
seq_start_loc=seq_start_loc, # if the sequence length is [4, 6], it is [0, 4, 10]. 这个是sequence
context_lens_tensor=context_lens_tensor, # context len的tensor，cache decode中不存储
block_tables=block_tables, # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
use_cuda_graph=use_captured_graph,
```

测试中的数据

```python
# 第一次 两个484传进去，prefill第一个484，第二个chunk28
num_prefills=2, num_prefill_tokens=512, num_decode_tokens=0, seq_lens=[28, 484], seq_lens_tensor=tensor([ 28, 484], device='cuda:0', dtype=torch.int32), max_query_len=484, max_prefill_seq_len=484, max_decode_seq_len=0, query_start_loc=tensor([  0,  28, 512], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0,  28, 512], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([0, 0], device='cuda:0', dtype=torch.int32), 
# 第二次，第一个推理485，第二个推理剩下的prefill
num_prefills=1, num_prefill_tokens=456, num_decode_tokens=1, seq_lens=[485, 484], seq_lens_tensor=tensor([485, 484], device='cuda:0', dtype=torch.int32), max_query_len=456, max_prefill_seq_len=484, max_decode_seq_len=485, query_start_loc=tensor([  0,   1, 457], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 485, 969], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([484,  28], device='cuda:0', dtype=torch.int32),
# 第三次 两个decode
num_prefills=0, num_prefill_tokens=0, num_decode_tokens=2, slot_mapping=tensor([2051253, 2050756], device='cuda:0'), seq_lens=[486, 485], seq_lens_tensor=tensor([486, 485], device='cuda:0', dtype=torch.int32), max_query_len=1, max_prefill_seq_len=0, max_decode_seq_len=486, query_start_loc=tensor([0, 1, 2], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 486, 971], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([485, 484], device='cuda:0', dtype=torch.int32)
```

如果对应一个剩下的prefill，block table照旧，slot mapping代表需要传进去的数据。



forward:

```

# decode部分
decode_query： [query decode的token数量, num_heads, head_size]
key_cache： [总的num_blocks, block_size, num_heads, head_size]
value_cache： [总的num_blocks, block_size, num_heads, head_size]
```

