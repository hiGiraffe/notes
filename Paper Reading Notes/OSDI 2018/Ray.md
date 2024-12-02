# Ray: A Distributed Framework for Emerging AI Applications

> 作者信息：
>
> UCB Ion Stoica实验室
>
> 链接：[[1712.05889] Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/abs/1712.05889)

## 一句话总结概括

运用于RL的通用集群计算框架

## 创新点或贡献

- To support workloads, 在engine之上统一了actor和task-parallel abstractions的抽象
- To achieve scalability and fault tolerance, 提出了一种系统设计原则
  - control state is stored in a sharded metadata store and all other system components are stateless.
- 至下而上的分布式调度策略

## 具体设计

- 统一的接口
  - 支持*task-parallel* and *actor-based* computations的表达

- Tasks
  - 提供给无状态的计算
  - 高效且动态的负载均衡
  - 故障恢复
- Actors
  - 支持有状态的计算（如训练）
  - 提供handle给其他actors或tasks

### Programming Model and Computation Model

**Programming Model**

1. Tasks 相关

```
futures = f.remote(args)
```

- Execute function *f* remotely
- Inputs: objects or future
- Output: one or more futures
- Non-blocking

```
objects = ray.get(futures)
```

- Return the values associated with one or more futures
- Blocking

```
ready_futures = ray.wait(futures, k, timeout)
```

- Return the futures whose corresponding tasks have completed as soon as either

  *k* have completed or the timeout expires.

2. Actors相关

```
actor = Class.remote(args)
```

- Instantiate class *Class* as a remote actor, and return a handle to it
- Non-blocking

```
futures = actor.method.remote(args)
```

- Call a method *futures* = *actor**.***method***.***remote**(*args*) on the remote actor and return one or more futures
- Non-blocking

---

**Computation Model**

动态任务计算图中有两种节点

1. Data objects and remote function invocations
2. Tasks

有三种边（有向边）

1. Data edges: capture the dependencies between data objects and tasks
2. Control edges: capture the computation dependencies that result from nested remote functions
3. Stateful edges: 表示有状态的依赖关系

### Architecture

![image-20241122113720266](Ray.assets/image-20241122113720266.png)

**Application Layer**

- Driver
  - 执行用户程序
- Worker
  - 执行无状态计算。自动分配任务
  - 被Driver或Worker启动
  - 没有需要跨Tasks维护的本地状态
- Actor
  - 有状态进程，仅执行其暴露的方法

---

**System Layer**

1. **Global Control Store (GCS)**

不太相关

---

2. **Bottom-Up Distributed Scheduler**

![image-20241122140758693](Ray.assets/image-20241122140758693.png)

为了避免全局调度器负荷太大

1. 先在Local Scheduler进行调度
2. 假如Local Scheduler服务超过了阈值或该节点资源不够，再转发给Global Scheduler

> 提升调度扩展性有几种方式： 1）批量调度。调度器批量提交任务给worker节点，以摊销提交任务带来的固定开销。Drizzle框架实现的就是这种。 2）层次调度。即全局调度器(global scheduler)将任务图划分到各个节点的本地调度器(local scheduler)。Canary框架实现了这种调度。 3）并行调度。多个全局调度器同时进行任务调度。这是Sparrow框架所做的。
>
> 但是他们都有各自的缺陷。 批量调度仍然需要一个全局调度器来处理所有任务。 层次调度假设任务图是已知的，即假设任务图是静态的。 并行调度假设每个全局调度器调度独立的作业。
>
> Ray希望做到的是高可扩展性，处理动态任务图，并且可能处理来自同一个作业的任务。
>
> 在框架设计上，local scheduler每隔一段时间会发送心跳包给**GCS**，注意不是直接发送给global scheduler，心跳包中会包含local scheduler的负载信息，GCS收到以后记录此信息，转发给global scheduler。 当收到local scheduler转发来的任务时，global scheduler使用最新的负载信息，以及人物的输入数据对象的位置和大小，来决定将task分发到哪个节点去运行。
>
> 如果global scheduler成为了瓶颈，那么采用多个副本，local scheduler随机选择一个global scheduler去转发任务。

---

Ray实现了存储信息和调度器的结构，使得系统有更多的可拓展性

## 实验评估



## 背景

### 先前工作存在的问题概述

在RL中，Training，Serving和Simulation都是耦合的

### 难点

完成一个分布式RL框架

- Fine-grained, heterogeneous computations.
  - 计算持续时间
  - 硬件异构
- Flexible computation model
  - 无状态
    - 可以在任意节点上执行，便于负载均衡和数据移动
  - 有状态
- Dynamic execution
  - 完成时间是不能提前知道的
  - 收敛次数也是不能提前知道的

### 补充背景 



## 思考角度

### 我如何做这个问题



### 这个洞见可以引申出其他其他方法吗

调度器结构和GSC的方式将复杂的调度问题解耦了，在别的调度问题中我们是否也可以这么解耦

### 该洞见是否可以迁移到其他领域中



## Q&A