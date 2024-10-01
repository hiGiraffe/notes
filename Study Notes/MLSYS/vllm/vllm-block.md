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

---

# 逻辑块与物理块

逻辑块保存在Sequence里面，且逻辑块在初始化的时候就定义好了

```
class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """
    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        self.token_ids = [_BLANK_TOKEN_ID] * block_size
        self.num_tokens = 0
```

虚拟block是记录有多少个token， token id是什么

```
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME

        self.computed = False #为prefix caching使用的
```

物理block则是管理kv cache

---

sequence类中的_append_tokens_to_blocks： 将token记录到block中，涉及新开一个虚拟block的机制。这部分是在推理一个token结束后触发的。

同时，append slot中会有_maybe_promote_last_block机制。会检查物理block和虚拟block的数量差距，检查是否要新开一个block。

* 假如不用新开一个block的情况下
  * 做prefix caching需要动那个物理block
  * 有涉及copy and write的机制需要动那个物理block
  * 否则也不动最后一个物理物理block

vLLM先给scheduler分配逻辑块，然后在append slot的时候会检查物理块和逻辑块的数量差距，加入物理块+1=逻辑块，就开多一个新的物理块。