# vLLM Ray

有关Ray的逻辑首先在llm_engine中的from_engine_args进行定义

1. 首先获取engine_config
2. 然后判断到是raygpu_executor，进行ray cluster初始化(executor/ray_tuils)
3. 然后定义RayGPUExecutor

[vllm代码走读（三）--executor(分布式) - 知乎](https://zhuanlan.zhihu.com/p/701992511)

[Transformer第九章：vllm并行化/分布式配置parallel\_config - 知乎](https://zhuanlan.zhihu.com/p/671660453)

[Ray分布式计算框架详解 - 知乎](https://zhuanlan.zhihu.com/p/460600694)