# vLLM cache



cache获取逻辑为：

cache engine -> backend -> paged_attention

---

**Cache_Engine部分**

对于gpu，cache_engine有gpu_engine和cpu_engine。

* 大小一致，为``List[torch.Tensor]``。
* List的序号代表``num_layers``。
* Tensor的大小为``kv_cache_shape``，在paged_attention中获得，shape:``(2, num_blocks, block_size * num_kv_heads * head_size)``

---

**Backend部分**

``query: shape = [num_tokens, num_heads \* head_size]``
``key: shape = [num_tokens, num_kv_heads \* head_size]``
``value: shape = [num_tokens, num_kv_heads \* head_size]``
``kv_cache = [2, num_blocks, block_size \* num_kv_heads \* head_size]``


copy的细节：

