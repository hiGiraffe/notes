# vLLM Ray

有关Ray的逻辑首先在llm_engine中的from_engine_args进行定义

1. 首先获取engine_config
2. 然后判断到是raygpu_executor，进行ray cluster初始化(executor/ray_tuils)
3. 然后定义RayGPUExecutor

https://zhuanlan.zhihu.com/p/701992511