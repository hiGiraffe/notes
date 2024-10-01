# LLM计算量统计——性能分析

[LLM Visualization](https://bbycroft.net/llm)

[LLM 推理阶段显存占用的学习与实践 - 知乎](https://zhuanlan.zhihu.com/p/713516682?utm_psn=1806366865662029825&utm_id=0)

[Transformer性能分析理论基础](https://github.com/HarleysZhang/dl_note/blob/main/6-llm_note/llm_inference/Transformer%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md)

该blog主要用来便于对LLM的计算量进行分析，从而选取合适的优化角度。



> [请问大模型在GPU进行上的推理时，核心计算是使用的tensor core 还是cuda core？ - 知乎](https://www.zhihu.com/question/636533414/answer/3577468768)
>
> 答案是既有tensor core又有cuda core，下面详细分析一下什么时候用tensor core，什么时候用cuda core。
>
> 首先大模型推理有如下几种类型的算子 Attention、FFN阶段的矩阵乘、RoPE、LayerNorm等。大模型推理又分为prefill阶段和decode阶段。
>
> RoPE、LayerNorm等用不到矩阵乘法的采用的是cuda core。
>
> Attention比较特殊，prefill阶段采用的是FlashAttention，底层q乘k、qk的结果乘v都用了tensor core。
>
> Attention decode阶段的由于每条query是1个token，目前主流的优化技术是PagedAttention，暂时没有使用tensor core，使用的是cuda core，
>
> 目前对于一些共享前缀的优化，decode阶段进行了优化，采用 tensor core + cuda core的方式。

[Transformer中QKV的矩阵运算 - 知乎](https://zhuanlan.zhihu.com/p/699573342)

[【LLM指北】五、参数量、计算量FLOPS推导 - 知乎](https://zhuanlan.zhihu.com/p/676113501)

[LLM训练指南(二):模型参数、计算量、显存、计算时间计算 - 知乎](https://zhuanlan.zhihu.com/p/639872915)

[learn-nlp-with-transformers/docs/篇章2-Transformer相关原理/2.4-图解GPT.md at main · datawhalechina/learn-nlp-with-transformers](https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.4-%E5%9B%BE%E8%A7%A3GPT.md)