# vLLM

[vLLM Blog](https://docs.vllm.ai/en/latest/index.html)

[vllm代码思维导图总结](..\..\..\inference.pdf)

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

---

vLLM外部逻辑

![image-20240626155335259](C:\Data Files\github repo\github blog\gitbook\images\vllm\1.png)

使用offline的方法，则会调用llm.py的generate函数

1. 增加所有request
2. 调用_run_engine进行推理