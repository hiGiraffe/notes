# [Llumnix: Dynamic Scheduling for Large Language Model Serving](https://arxiv.org/abs/2406.03243)

Alibaba在OSDI 2024的一篇工作

> 作者信息

## 一句话总结概括



## 背景



## 先前工作存在的问题



## 难点



## 解决方案

![](C:\Users\user\Nutstore\1\Nutstore\gitbook\images\Llumnix\1.png)

1. 图a的负载均衡(*load balancing*)，通过减少请求的动态不确定的影响。但会带来新的问题： higher **memory fragmentation** and longer **queuing delays** of long inputs probably.
2. 图b的去碎片化(*de-fragmentation*)，通过去碎片化获得更完整的内存空间，使得长请求可以被调度。
3. 图c的优先级(*prioritization*) ，通过调度使得高优先级的请求获得更低的负载和更少的干扰，为高优先级请求保留了更多的资源。
4. 图d的自动缩放(*auto-scaling*)，这个没看懂是啥。to drain out an instance to be terminated(1-d) or saturate a new instance more quickly.

## 创新点



## 实验评估



## Q&A