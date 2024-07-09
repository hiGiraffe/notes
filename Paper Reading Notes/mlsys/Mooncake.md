# [Mooncake: Kimi’s KVCache-centric Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)

月之暗面kimi团队在2024的一篇工作

---

## Abstract

* 以KV cache为核心
* 采用了prefill和decode的分解架构
* 利用了集群中未被使用的CPU、DRAM和SSD等资源
* 在最大化吞吐量的同时，还平衡了服务级别目标
* 面临的是高度超载的环境，提出了基于预测的早期拒绝政策，



找到问题的根源，然后假如decode阶段越多越好，我们的preempt政策可能效果更好

## 背景

1. Model as a Service (MaaS)的最大目标是 maximize overall effective throughput和尽量约束varying levels of SLOs，比如 the time to first token (TTFT)和 the time between tokens (TBT)。
2. prefill和decoding计算特性不一样，前者是compute bound，后者是memory bound，曾有相关工作提出了分离架构。

## 存在问题



## 难点



## 解决方案



## 创新点

