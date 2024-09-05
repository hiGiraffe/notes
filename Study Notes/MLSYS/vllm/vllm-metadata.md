## Output机制

1. _schedule做的是，输出SchedulerOutputs

其中，调度结果包含

```python
# 参与调度所有seq
scheduled_seq_groups=(prefills.seq_groups + running_scheduled.decode_seq_groups  + swapped_in.decode_seq_groups),
# prefill的数量
num_prefill_groups=len(prefills.seq_groups),
# 当前batch的token数量
num_batched_tokens=budget.num_batched_tokens,
# physical block需要做的变化
blocks_to_swap_in=swapped_in.blocks_to_swap_in,
blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
blocks_to_copy=running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy,
# 被取消服务的seq
ignored_seq_groups=prefills.ignored_seq_groups + swapped_in.infeasible_seq_groups,
# lookahead，应该是speculative decoding那一块的lookahead decoding
num_lookahead_slots=running_scheduled.num_lookahead_slots,
# 在推理计算中的queue
running_queue_size=len(self.running),
# 被抢占的queue
preempted=preempted,
```

---

2. scheduler在获取schedule结果后，用SchedulerOutputs的结果生成seq_group_metadata_list

seq_group_metadata_list就是对_schedule中每一个schedule_seq_groups进行处理

```python
request_id=seq_group.request_id,
is_prompt=is_prompt,
# seq_data 就是req中seq_id -> SequenceData的字典
seq_data=seq_data,
sampling_params=seq_group.sampling_params,
# block_tables则是req中seq_id -> physical block numbers的字典
block_tables=block_tables,
# 假如seq在该prefill，tokens不能计算完（chunked），则设置为False；否则为True
do_sample=do_sample,
pooling_params=seq_group.pooling_params,
token_chunk_size=token_chunk_size,
lora_request=seq_group.lora_request,
# 从block_manager中获取该seq的common computed block ids
computed_block_nums=common_computed_block_nums,
state=seq_group.state,
multi_modal_data=seq_group.multi_modal_data
if scheduler_outputs.num_prefill_groups > 0 else None,
```

---

3. llm_engine将scheduler.schedule()生成的结果再次生成ExecuteModelRequest，如何传入execute_model进行计算。其实发现就五个数据会传入execute_model中进行计算。

```python
seq_group_metadata_list=seq_group_metadata_list,
blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
blocks_to_copy=scheduler_outputs.blocks_to_copy,
num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
running_queue_size=scheduler_outputs.running_queue_size,
```

---

4. 一路传递下去，在worker阶段

blocks_to_swap_in，blocks_to_swap_out和blocks_to_copy会进行cache_swap

seq_group_metadata_list和kv_cache会传入model_runner进行计算

---

5. model runner阶段

在推理前会调用prepare_input_tensors将seq_group_metadata_list转化input_tokens, input_positions, attn_metadata, sampling_metadata, lora_requests, lora_mapping, multi_modal_input

```
input_tokens, 
input_positions, 
attn_metadata,
sampling_metadata, 
lora_requests, 
lora_mapping,
multi_modal_input
```

prepare_input_tensors会继续调用_prepare_model_input来处理seq_group_metadata_list信息

```python
input_tokens=input_tokens_tensor,
input_positions=input_positions_tensor,
attn_metadata=attn_metadata,
seq_lens=seq_lens,
query_lens=query_lens,
lora_mapping=lora_mapping,
lora_requests=lora_requests,
multi_modal_input=multi_modal_input,
slot_mapping=slot_mapping_tensor,
num_prefill_tokens=num_prefill_tokens,
num_decode_tokens=num_decode_tokens,
num_prefills=num_prefills,
```

---

6. 运行推理

model_runner向model传入execute_model_kwargs

```
"input_ids": input_tokens,
"positions": input_positions,
"kv_caches": kv_caches,
"attn_metadata": attn_metadata,
```

---

7. opt

decoder层

1. embed_tokens层使用input_ids，有需要project_in则传入project_in层
2. embed_positions层使用positions
3. hidden_states=inputs_embeds + pos_embeds
4. 对于每一层，传入hidden_states、对应层的kv_caches和attn_metadata
   1. self_attn_layer_norm层使用hidden_states
   2. self_attn层使用hidden_states、kv_cache、attn_metadata
      1. qkv层使用hidden_states
      2. attn层使用q、k、v、kv_cache和attn_metadata
      3. out_proj层使用hidden_states
   3. hidden_states = residual + hidden_states（residual 是attn前的hidden_state）
   4. final_layer_norm层使用hidden_states
   5. fc1层使用hidden_states
   6. activation_fn层使用hidden_states
   7. fc2层使用hidden_states
   8. hidden_states = residual + hidden_states（residual 是mlp前的hidden_state）
5. 计算结果传入final_layer_norm和project_out

---

8. sampling_metadata

[Sampling Parameters — vLLM](https://docs.vllm.ai/en/latest/dev/sampling_params.html)

传进去prepare函数

```
seq_group_metadata_list: List[SequenceGroupMetadata],
seq_lens: List[int],
query_lens: Optional[List[int]],
device: str,
pin_memory: bool,
```

先根据这些数据生成

[vLLM（六）源码解读下 - 知乎](https://zhuanlan.zhihu.com/p/694442998)

[vllm代码走读(六）--后处理 - 知乎](https://zhuanlan.zhihu.com/p/707698441)

```
 self.seq_groups = seq_groups
 self.selected_token_indices = selected_token_indices
 self.categorized_sample_indices = categorized_sample_indices
 self.num_prompts = num_prompts
```



