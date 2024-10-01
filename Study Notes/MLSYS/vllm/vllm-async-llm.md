# vLLM async相关逻辑

便于开发和调试，这里只涉及vLLM的api_server，不涉及OpenAI的API

## API命令

```
python -m vllm.entrypoints.openai.api_server \
 --model facebook/opt-125m \
 --tensor-parallel-size 1 \
 --max-num-seqs 500 \
 --trust-remote-code \
 --enforce-eager \
 --max-num-batched-tokens 512 \
 --enable-chunked-prefill 
 
 
 --gpu-memory-utilization 0.02 \
 
```

```
python ~/coserving/entrypoints/openai/api_server.py \
 --model facebook/opt-125m \
 --tensor-parallel-size 1 \
 --max-num-seqs 500 \
 --trust-remote-code \
 --enforce-eager \
 --max-num-batched-tokens 512 \
 --enable-chunked-prefill 
```



Server

```
python entrypoints/openai/api_server.py \
 --model facebook/opt-125m \
 --tensor-parallel-size 1 \
 --max-num-seqs 500 \
 --trust-remote-code \
 --enforce-eager \
 --max-num-batched-tokens 512 \
 --enable-chunked-prefill 
```

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "facebook/opt-125m",
     "messages": [{"role": "user", "content": "1 u are loved, sir: They that least lend it you shall lack you first. KING I fill a place"}],
     "temperature": 0.7,
     "length_penalty": 1.0,
     "max_tokens": 100,
     "min_tokens": 100,
     "is_latency_sensitive": "True",
     "slo":0.05
   }'
```



## 代码

目前先考虑异步的api server，OpenAI的API相关逻辑后续再讨论

---

* api_server，触发generate函数(async def generate)
  - engine.generate
  - async for request_output in results_generator:
    - 如果disconnected
      - abort该req
    - final_output = request_output

简而言之，api server可以在每一次iteration后异步获取新的结果

---

* AsyncLLMEngine类

  * generate函数
    
    * 异步函数，调用_process_request计算output，每个iteration会返回新的结果
  * _process_request函数
    
    * 异步函数，调用self.add_request添加req，然后每次有新的结果就返回
  * add_request函数
    * 异步函数，返回AsyncStream，记录每次结果
    * 假如没开启backgroud loop，则调用start_background_loop来使得后端运行
    * 调用self.engine.process_model_inputs_async函数来进行计算
  * 调用self._request_tracker.add_request更新stream
  
* 返回stream
  
  * start_background_loop是关键，启动后台的循环处理run_engine_loop
  
    ```python
        def start_background_loop(self) -> None:
            """Start the background loop."""
            if self.errored: # 检查错误
                raise AsyncEngineDeadError(
                    "Background loop has errored already.") from self._errored_with
            if self.is_running: # 检查错误
                raise RuntimeError("Background loop is already running.")
            # Initialize the RequestTracker here so it uses the right event loop.
            self._request_tracker = RequestTracker() # 初始化tracker
    		
            # 获取一个循环函数run_engine_loop
            self._background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_engine_loop())
            # 创建回调函数
            self._background_loop_unshielded.add_done_callback(
                partial(_raise_exception_on_finish,
                        error_callback=self._error_callback))
            # 保护协程
          self.background_loop = asyncio.shield(self._background_loop_unshielded)
    ```
  
  * run_engine_loop
    
    * while循环调用engine_step
  * engine_step
    * 获取新的req和已完成的req
      * 新的req加入到engine中
      * 已完成的req就abort掉
    * 根据是否使用ray，运行step.remote()或step_async()
    * 调用process_request_output处理输出

简而言之，这部分通过add request，然后异步地获取其输出，有输出就激活await，继续操作。写的比较巧妙，对于不熟悉异步逻辑的人来说还是有点难度的。