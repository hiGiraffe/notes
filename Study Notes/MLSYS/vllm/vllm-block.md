# Detailed explanation of vllm block mechanism

## 逻辑块







---

BlockManager这个class下又维护着两个重要属性：

- **`BlockAllocator`：物理块分配者，负责实际为seq做物理块的分配、释放、拷贝等操作。**其下又分成`self.gpu_allocator`和`self.cpu_allocator`两种类型，分别管理gpu和cpu上的物理块。

- **`self.block_tables`：负责维护每个seq下的物理块列表，本质上它是一个字典，形式如`{seq_id: List[PhysicalTokenBlock]}`。**注意，这个字典维护着【所有】seq_group下seq的物理块，而不是单独某一个seq的。因为调度器是全局的，所以它下面的的BlockManager自然也是全局的。

**其中，BlockAllocator又分成两种类型：**

- **`CachedBlockAllocator`**：**按照prefix caching的思想来分配和管理物理块**。在原理篇中，我们提过又些prompts中可能含有类似system message（例如，“假设你是一个能提供帮助的行车导航”）E）等prefix信息，带有这些相同prefix信息的prompt完全可以共享用于存放prefix的物理块，这样既节省显存，也不用再对prefix做推理。
- **`UncachedBlockAllocator`**：**正常分配和管理物理块，没有额外实现prefix caching的功能**。





## UncachedBlockAllocator

**在vllm的1个推理阶段，所有的seq_group要么一起做prefill，要么一起做decode**。

其中，

* waiting：等待做prefill的
* running/running+swapped：等待做decode的

### Block_Manager_v1

* allocate(): 为当前seq_group分配物理块做**prefill**
  * 假如是UncachedBlockAllocator
    * allocated一个free物理块。
    * 将其ref_count设置为num_seqs，表示有num_seqs个逻辑块引用这个物理课。
    * 将这个物理块加入block_table

* append_slots()：为running/swapped队列中的seq_group分配物理块做**decode**
  * 如果物理块< 逻辑块，分配一个，退出
  * 如果最后一个物理块只被一个逻辑块引用（必须是gpu物理块，可能是prefix caching）
    * 使用prefix caching……，退出
    * 不适用prefix caching，退出
  * 如果最后一个物理块被多个逻辑块引用
    * 触发copy-on-write机制
    * 新开一个物理块
    * 释放掉旧的物理块
    * 记录到seq的block_table中

