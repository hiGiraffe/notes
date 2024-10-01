# 【学习笔记】大模型训练：流水线并行

[原文链接](https://zhuanlan.zhihu.com/p/613196255)


> 经典的流水线并行范式有Google推出的**Gpipe**，和微软推出的**PipeDream**。两者的推出时间都在2019年左右，大体设计框架一致。主要差别为：在梯度更新上，Gpipe是同步的，PipeDream是异步的。异步方法更进一步降低了GPU的空转时间比。虽然PipeDream设计更精妙些，但是Gpipe因为其“够用”和浅显易懂，更受大众欢迎（torch的pp接口就基于Gpipe）。因此本文以Gpipe作为流水线并行的范例进行介绍。



分布式训练的总体目标：

* 训练更大的模型
* 更快地训练模型

难点：

* 模型参数和中间结果更多，内存压力大
* GPU之间的传输增大，通信开销大

## 模型并行

将模型隔成不同的层，每一层放到一块GPU上

>  ![img](/images/llm-1/1)

此时模型前向传输和后向传输

>  ![img](/images/llm-1/2)

> 其中下标表示batch编号，这里只有一个batch，因此下标都是0。每一行表示一个GPU。每一列表示timestep。
>
> 这张图的含义是：我在GPU0上做完一次forward，然后将GPU0上最后一层的输入传给GPU1，继续做forward，直到四块GPU都做完forward后，我再依次做backward。等把四块GPU上的backward全部做完后，最后一个时刻我统一更新每一层的梯度。

这样会带来以下问题：

* GPU利用度不够
* 中间结果占据大量内存

## 流水线并行

针对上述问题，Gpipe提出了流水线并行。

### 1.切分micro-batch

> **在模型并行的基础上，进一步引入数据并行的办法，即把原先的数据再划分成若干个batch，送入GPU进行训练**。未划分前的数据，叫**mini-batch**。在mini-batch上再划分的数据，叫**micro-batch**。
>
> ![img](/images/llm-1/3)

### 2.re-materialization（active checkpoint）

> Gpipe采用了一种非常简单粗暴但有效的办法：**用时间换空间，在论文里，这种方法被命名为re-materalization，后人也称其为active checkpoint**。
> 具体来说，就是**几乎不存中间结果，等到backward的时候，再重新算一遍forward**
>
> ![img](/images/llm-1/4)
>
> 每块GPU上，我们只保存来自上一块的最后一层输入z，其余的中间结果我们算完就废。等到backward的时候再由保存下来的z重新进行forward来算出。
>
> 如果你使用Pytorch提供的pipeline接口，其中有一个参数叫checkpoint，就是用来做这一项的。
>
> 在micro-batch的划分下，我们在计算**Batch Normalization**时会有影响。Gpipe的方法是，在训练时计算和运用的是micro-batch里的均值和方差，但同时持续追踪全部mini-batch的移动平均和方差，以便在测试阶段进行使用。Layer Normalization则不受影响。



### Todo

[pytorch流水线并行源码解析](https://zhuanlan.zhihu.com/p/665993846)
