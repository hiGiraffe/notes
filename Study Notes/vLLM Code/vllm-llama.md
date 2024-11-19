# vLLM Llama部分的设置

https://cloud.tencent.com/developer/article/2314990 Rotary Embedding

https://www.cnblogs.com/rossiXYZ/p/15871062.html 并行MLP



loader机制

 ```
1 	/home/cjl/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6
2 	['/home/cjl/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/pytorch_model.bin']
3 	False
4 	facebook/opt-125m
5 	None
6 	True
 ```

```
1 	print(hf_folder)
2	print(hf_weights_files)
3	print(use_safetensors)
4	print(model_name_or_path)
5	print(revision)
6	print(fall_back_to_pt)
```

