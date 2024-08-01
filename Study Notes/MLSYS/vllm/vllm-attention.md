# vLLM Paged Attention

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
