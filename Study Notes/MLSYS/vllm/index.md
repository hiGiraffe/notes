# vLLM

[vLLM Blog](https://docs.vllm.ai/en/latest/index.html)

---

vLLM代码架构：

* **attention:** paged attention相关，主要涉及backend、paged attention、flash attention和prefix_prefill
* **core：**调度机制相关，主要涉及scheduler、block_manager和policy
* **distributed：**分布式相关
* **engine：**api相关，离线+在线
* **entrypoints：**api和后端对接相关，llm和api_server
* **executor**
* **lora**
* **model_executor**
* **spec_decode**
* **transformers_utils**
* **usage**
* **worker**