# vLLM chunked prefill机制

**Scheduler**

1. schedule as many decoding requests as possible. 
2. schedule chunked prefill requests that are not finished.
3. schedule swapped request. 
4. schedule new prefill requests.

---

调度机制

通过_get_num_new_tokens获取可以支持的tokens数目。

chunked prefill调度机制

> 1. 先调度running
> 2. 再调度swapped
> 3. 再调度prefills

default调度机制

> 1. 假如没有swapped，则先调度prefill
> 2. 假如没有prefill，则调度running
> 3. 再调度swapped

* 在schedule_running阶段，通过prefill_seq_groups进行管理，其实就是通过ScheduledSequenceGroup的token_chunk_size来控制一次inference的token数量


---

如何记录chunk prefill是到哪一个token

* 在llm_emgine中有_process_model_outputs函数，会进一步调用seq的update_num_computed_tokens来更新tokens。

* 在model_runner中，在推理前有_prepare_model_input，会更新调度的token num来准备input的数据。

* 在Sequence中，维护了一个computed_tokens