# vLLM Schedule

[vLLM arxiv论文](https://arxiv.org/abs/2309.06180)

[vLLM关于PagedAttention的博客](https://blog.vllm.ai/2023/06/20/vllm.html)

[vLLM官方文档](https://docs.vllm.ai/en/latest/)

[国内博客](https://readpaper.feishu.cn/docx/EcZxdsf4uozCoixdU3NcW03snwV)

# 源码解读

[源码笔记](https://github.com/hiGiraffe/vllm)

[图解大模型计算加速系列：vLLM源码解析1，整体架构](https://zhuanlan.zhihu.com/p/691045737)

[图解大模型计算加速系列：vLLM源码解析2，调度器策略(Scheduler)](https://zhuanlan.zhihu.com/p/692540949)

> 在vLLM中，当我们使用离线批处理模式时，表面上是在做“同步”推理，也即batch_size是静态固定的。**但推理内核引擎（LLMEngine）在实际运作时，batch_size是可以动态变更的**：在每一个推理阶段（**prefill算1个推理阶段，每个decode各算1个推理阶段**）处理的batch size可以根据当下显存的实际使用情况而变动。
>
> 
> 举个例子来说：
>
> - 给定一个很大的batch，此时尽管vLLM采用了PagedAttention这样的显存优化技术，我们的gpu依然无法同时处理这么大的batch。
> - 所以batch中的每一条数据，会被先放到一个waiting队列中。vLLM会用自己的调度策略从waiting队列中依次取数，加入running队列中，直到它认为取出的这些数据将会打满它为1个推理阶段分配好的显存。此时waiting队列中可能还会剩一些数据。
> - 在每1个推理阶段，vLLM对running队列中的数据做推理。如果这1个推理阶段执行完毕后，有的数据已经完成了生成（比如正常遇到`<eos>`了），就将这些完成的数据从running队列中移开，并释放它占据的物理块显存。
> - 这时，waiting队列中的数据就可以继续append进running队列中，做下1个阶段的推理。
> - 因此在每1个推理阶段，vLLM处理的batch size可能会动态变更。
> - 将LLMEngine包装成离线批处理形式后，所有的数据必须等到一起做完推理才能返给我们。所以从体感上，我们可能很难感知到内核引擎的“动态”逻辑。
>
> **在vLLM中，即使是同步形式的离线批处理，其背后的内核引擎也是按动态batch的形式来实现的**
>
> **正是因为LLMEngine这种“动态处理”的特性，才使得它同时也能成为异步在线服务的内核引擎**：当一条条请求发来时，它们都先进入LLMEngine调度器（Scheduler）的waiting队列中（实际并不是直接进入waiting队列中的，而是在传给LLMEngine前先进入asyncio.Queue()中，然后再由LLMEngine调度进waiting队列中的，这些细节我们也放在后面说，这里不影响理解就行）。此时模型正常执行它的1个推理阶段，调度器也正常处理新来的请求。当模型准备执行下1个推理阶段时，调度器再根据设定的策略，决定哪些数据可以进入running队列进行推理。由于在线服务是异步的，先推理完成的数据就可以先发给客户端了（如果采用流式传输，也可以生成多少先发多少）。
> **在这个过程中，vLLM通过PagedAttention技术和“先来先服务（FCFS），后来先抢占，gpu不够就先swap到cpu上”的调度策略，在1个推理阶段处理尽可能多的请求，解决高并发场景下的推理吞吐问题。这就是整个vLLM运作的核心思想。（对这行黑体字里的术语有疑惑的朋友，建议先看vLLM原理篇讲解）**

> ![img](/images/llm-4/12)



> 先来看**`LLMEngine`**：
>
> - **`add_request()`**：该方法将每一个请求包装成vLLM能处理的数据类型(SequenceGroup，后面我们会详细解释)，并将其加入调度器（Scheduler）的waiting队列中。**在LLMEngine中，这个函数是按照“同步”的方式设计的**，也就是它被设计为“遍历batch中的每条数据，然后做相应处理”。所以这个函数本身只适合批处理场景。在异步的online serving中将会把它重写成异步的形式。
> - **`abort_request`**：在推理过程中，并不是所有的请求都能有返回结果。比如客户端断开连接时，这个请求的推理就可以终止了（abort），这个函数就被用来做这个操作。
> - **`step()`：负责执行1次推理过程（1个prefill算1个次推理，每个decode各算1次推理）**。在这个函数中，vLLM的调度器会决定要送那些数据去执行本次推理，并负责给这些数据分配好物理块（这些信息都被作为metadata放在要送给模型做推理的数据中）。模型会根据这些信息，采用PagedAttention方法，实际完成推理。

## 整体代码架构

> ![img](/images/llm-4/13)
>
> ### Centralized Controller
>
> **Centralized Controller，也就是前文我们所说的调度器(Scheduler)**。它和LLMEngine所在的进程是同一个，且两者都是在CPU上的。
>
> - **调度器的主要作用就是，在每1个推理阶段，决定要把哪些数据送给模型做推理，同时负责给这些模型分配KV Cache物理块**。但要注意，它只是分配了物理块的id，而不是物理块本身。物理块的实际分配是模型在推理过程中根据物理块id来操作的，也就是CacheEngine做的事情。
> - **调度器下维护着BlockSpaceManager。它负责管理BlockAllocator（实际参与分配物理块的类）。BlockAllocator又分成gpu和cpu两种类型，分别管理这两类设备上的物理块**。**你可能会问，cpu上的物理块是什么呢**？你还记得调度器有一个swap策略吗？当gpu上显存不足时，它会把后来的请求抢占，并将其相关的KV cache物理块全部都先swap（置换、卸载）在cpu上，等后续gpu显存充足时，再把它们加载回gpu上继续做相关请求的推理。所以在cpu上我们也需要一个管控物理块的BlockAllocator。**实际代码实现时，Block相关的部分可不止这两个class，还有一些更复杂的逻辑细节。这个我们放在本系列后面的文章中讲解**。
>
> ### Distributed Workers
>
> Distributed Workers，也就是分布式系统，你可以将每个worker理解成一块gpu。它的作用是将我们要使用的模型load到各块卡上（目前对单卡装不下的模型，vLLM支持tp/pp推理），然后对Controller传来的数据做1次推理，返回相关结果。我们来细看下这块：
>
> - **Distributed Workers**：图中绘制为Distributed Workers这个绿色块，**其实按vLLM的源码内容，写成Executor会更合适一些**。**它就是所有Workers的管控中心**，它指定了用什么方法管控这些Workers，负责分布式环境的初始化，目前支持的方法有：
>
> - - cpu_executor：（较少用），使用cpu做推理时可考虑
>   - gpu_executor：单卡（world_size = 1）的情况下可用
>   - ray_gpu_executor：使用ray这个分布式计算框架实现的executor，适用于多卡环境
>
> - **Worker**：**在硬件上，它指gpu；在代码上，它指的是Worker实例（每个gpu上的进程维护自己的Worker实例）**。在每个Worker实例中又管控着如下两个重要实例：
>
> - - **CacheEngine：**负责管控gpu/cpu上的KV cache物理块（调度器的block manager只负责物理块id的分配，CacheEngine则是根据这个id分配结果实打实地在管理物理块中的数据）
>   - **Worker.model**：根据vLLM代码，这里写成**model_runner**会更合适一些。**它负责加载模型，并执行推理**。PagedAttention的相关逻辑，就维护这个实例关联的代码下。

## 运行逻辑

> **在vLLM正式开始处理1条请求（也就是LLMEngine的调度器正式开始运作时），它需要做两件和初始化相关的事：**
>
> - **加载模型**
> - **预分配显存**
>
> 1. 模型加载
>
> ![img](/images/llm-4/14)
>
> 这里在做的事很直观：把你的base model加载到worker上。如果你是online加载的，vLLM默认使用HuggingFace，你也可以在环境变量中把相关配置改成ModelScope。
>
> 2. 预分配显存
>
>    ![img](/images/llm-4/15)
>
>    **在模型部署的初始化阶段（推理正式开始前），vLLM会通过模拟实验的方式，来决定gpu/cpu上到底有多少个KV cache物理块可以分配给后续的请求们做推理。vLLM管这个步骤叫`determine_num_available_blocks`，跟文章中的不一样**
>
>    **（1）杜撰假数据**
>
>    
>    **（2）用假数据模拟一次前向推理**
>
>    **我们现在想知道在1次推理过程中，可以分配多少的显存给KV cache。我们可以使用如下公式计算：**
>    **分配给KV cache显存 = gpu总显存 - 不使用KV cache做1次推理时的显存占用（包括模型本身和推理过程中的中间数据）**
>
>    对于“不使用KV cache做1次推理时的显存占用”，我们就可以用杜撰出来的假数据模拟一次前向推理来计算得出。在前向推理之后，我们把gpu上的缓存清一次，让它不要影响后续模型的正常推理。
>
>    **（3）计算可分配的KV cache物理块总数**
>
>    CPU上物理块总数也是同理，但与GPU不同的是，它不需要做模拟实验。CPU上可用的内存总数是用户通过参数传进来的（默认是4G）。也就是我们认为只能在这4G的空间上做swap。将上面公式中“分配给KV Cache的显存大小”替换成4G，就能得到CPU上物理块的数量。
>
>    **（4）将预分配的KV Cache加载到gpu上**
>
>    **当我们确定好KV Cache block的大小后，我们就可以创建empty tensor，将其先放置到gpu上，实现显存的预分配。以后这块显存就是专门用来做KV Cache的了。**也正是因为这种预分配，你可能会发现在vLLM初始化后，显存的占用比你预想地要多（高过模型大小），这就是预分配起的作用。相关代码如下（帮助大家更好看一下KV cache tensor的shape）:

## Scheduler调度

> ![img](/images/llm-4/16)
>
> ![img](/images/llm-4/17)
>
> **vLLM的调度策略中有一项叫做：后来先抢占（\*Preemption\*）**。它是指在准备执行当前这1个推理阶段时，如果gpu上没有足够的资源对running队列中的全部数据完成下1次推理，我们就取出running队列中最后来的数据，将它的KV Cache swapped到CPU上，同时将这个数据从running移到swapped中。**我们重复执行这个步骤，直到当前gpu上有足够的KV Cache空间留给剩在running中的全部数据为止。**

## LLM函数

> 当我们调用·`outputs = llm.generate(prompts, sampling_params)`时，**它实际做了两件事情：**
>
> - **`_add_request`**：**将输入数据传给LLMEngine**，它具体做了如下事情：
>
> - - **把每1个prompt包装成一个SequenceGroup对象**。从客户端角度看，1个请求可能包含多个prompts，例如离线批处理场景下你可以将1个batch理解成1个请求；但是从LLMEngine的角度看，1个prompt是1个请求，所以它会对输入数据进行预处理。在后文对SequenceGroup的讲解中，我们会来看vLLM这样做的意义。
>   - **把包装成SequenceGroup对象的数据加入调度器（Scheduler）的waiting队列，等待处理**。这一块相关的细节，我们放在后文说。
>
> 
>
> - **`_run_engine`**：**执行推理**。只要调度器的waiting/running/swapped队列非空，我们就认为此时这批batch还没有做完推理，这时我们就会**调用LLMEngine的step()**函数，来完成1次调度以决定要送哪些数据去做推理。
>
>
> **所以，想要知道调度器的运作流程，我们只要从`LLMEngine`的`add_request()`和`step()`两个函数入手就好了**。**不过在正式进入这两个函数的讲解之前，我们先来看和输入数据一个问题：为什么要把每个prompt都包装成一个SequenceGroup实例？SequenceGroup又长什么样呢？**

## SequenceGroup

> **可能出现"1个prompt -> 多个outputs"的情况。那是否能设计一种办法，对1个prompt下所有的outputs进行集中管理，来方便vLLM更好做推理呢？**

> ### SequenceGroup的作用 
>
> - **"1个prompt -> 多个outputs"这样的结构组成一个`SequenceGroup`实例。**
>
> - **其中每组"prompt -> output"组成一个序列（seq，属于`Sequence`实例），每个seq下有若干状态(status)属性，包括：**
>
> - - **`WAITING`：**正在waiting队列中。waiting队列中的序列都没有做过prefill。
>
>   - **`RUNNING`：**正在running队列中，即已经开始做推理。
>
>   - **`SWAPPED`：**正在swapped队列中，表示此时gpu资源不足，相关的seq_group被抢占，导致其暂停推理，相关的KV block被置换到cpu上（swap out），等待gpu资源充足时再置换回来重新计算（swap in）。
>
>   - **若干和Finish相关的状态**，表示该seq推理已经结束，具体包括：
>
>   - - **`FINISHED_STOPPED`：**正常执行完毕，例如碰到`<eos>`符号，该seq的推理正常结束了
>     - **`FINISHED_LENGTH_CAPPED`**：因为seq的长度达到最大长度限制，而结束推理
>     - **`FINISHED_ABORTED`**：因不正常状态，而被终止的推理。例如客户端断开连接，则服务器会终止相关seq的推理
>     - **`FINISHED_IGNORED`**：因prompt过长而被终止执行的推理。本质上也是受到长度限制
>
> - **在vLLM中有一个重要假设：一个seq_group中的所有seq共享1个prompt。**
>
> 例子：
>
> - **在推理开始之前**，这个seq_group下只有1条seq，它就是prompt，状态为waiting。
>
> - **在第1个推理阶段**，调度器选中了这个seq_group，由于它的采样参数中`n = 4`，所以在做完prefill之后，它会生成4个seq，它们的状态都是running。
>
> - **在若干个推理阶段后，gpu上的资源不够了，这个seq_group不幸被调度器抢占（preemption）**，它相关的KV block也被swap out到cpu上。此时所有seq的状态变为swapped。这里要注意，**当一个seq_group被抢占时，对它的处理有两种方式：**
>
> - - **Swap：如果该seq_group下的seq数量 > 1，此时会采取swap策略**，即把seq_group下【所有】seq的KV block从gpu上卸载到cpu上。（seq数量比较多，直接把算出的KV block抛弃，比较可惜）
>   - **Recomputation：如果该seq_group下的seq数量 = 1，此时会采取recomputation策略**，即把该seq_group相关的物理块都释放掉，然后将它重新放回waiting队列中。等下次它被选中推理时，就是从prefill阶段开始重新推理了，因此被称为“重计算”。（seq数量少，重新计算KV block的成本不高）
>
> **【注意，并不是每个seq_group都会经历抢占，具体要看调度器策略和gpu资源使用情况】**
>
> - **又过了若干个推理阶段，gpu上的资源又充足了，此时执行swap in操作**，将卸载到cpu上的KV block重新读到gpu上，继续对该seq_group做推理，此时seq的状态又变为running。
> - **又过了若干个推理阶段，该seq_group中有1个seq已经推理完成了，它的状态就被标记为finish**，此后这条已经完成的seq将不参与调度。
> - **又过了若干个推理阶段，这个seq_group下所有的seq都已经完成推理了**，这样就可以把它作为最终output返回了。
>
> ![img](/images/llm-4/18)

> **SequenceGroup:**
>
> - **`self.seqs_dict`**：{seq_id: seq}，其中每个seq是一个Sequence对象。正如我们前文介绍的那样，一个seq_group下包含若干seqs
> - **`self.sampling_params`**：采样参数
> - **`self.metrics`**：**记录该seq_group相关的指标，例如该seq_group是什么时候被加入LLMEngine的（arrival_time）**，该seq_group第一次被调度器选中调度是什么时候等等。调度器在选择时，会参考seq_groups们的这些指标来做决策。
> - **`get_max_num_running_steps`**：**该seq_group在剩余生命周期内并行running的最大seq数量**。**“剩余生命周期”指从此刻一直到seq_group中所有的seq都做完推理**。举个例子来说，我们看2.2节配图中倒数第3个时刻，此时这个seq_group内所有的seq都还没结束推理，所以若调用这个方法，则返回值为4；再看倒数第2个时刻，此时有1个seq已经完成了推理，所以若调用这个方法，则返回值为3。在后续调度策略代码中，我们将经常看到这个方法被调用，目的是用于估计若当前对一个seq_group做推理，它将消耗多少gpu资源。

> Sequence:
>
> 对于一个seq，我们重点来看它的属性`self.logical_token_blocks`（逻辑块）和方法`_append_tokens_to_blocks`（生成逻辑块的方法）。**在vLLM中，每个seq都单独维护一份属于自己的逻辑块，不同的逻辑块可以指向同一个物理块**（此刻你一定很关心逻辑块和物理块是如何做映射的，我们会循序渐进地讲解这点，**现在你可以先忽略映射方法，把目光聚焦于“一个seq的逻辑块长什么样，怎么初始化它的逻辑块”**）
>
> 分配 _append_tokens_to_blocks

## add_request()

>  将seq_group添加进调度器waiting队列

## step()

**调度器结构**

> ![img](/images/llm-4/19)
>
> - **`self.waiting, self.running, self.swapped`**：这三个都是python的deque()实例（双端队列，允许你从队列两侧添加或删除元素）。
>
> - - **waiting队列用于存放所有还未开始做推理的seq_group**，“未开始”指连prefill阶段都没有经历过。所以waiting队列中的seq_group只有一个seq，即是原始的prompt。
>   - **running队列用于存放当前正在做推理的seq_group。更准确地说，它存放的是上1个推理阶段被送去做推理的seq_group们**，在开始新一轮推理阶段时，调度器会根据本轮的筛选结果，更新running队列，即决定本轮要送哪些seq_group去做推理。
>   - **swapped队列用于存放被抢占的seq_group**。在2.2节中我们有提过，若一个seq_group被抢占，调度器会对它执行swap或recomputation操作，分别对应着将它送去swapped队列或waiting队列，在后文我们会详细分析抢占处理的代码
>
> - **`self.policy`：是vLLM自定义的一个Policy实例，**目标是根据调度器总策略（**FCFS**，First Come First Serve，先来先服务）原则，**对各个队列里的seq_group按照其arrival time进行排序**。相关代码比较好读，所以这里我们只概述它的作用，后续不再介绍它的代码实现。
> - **`self.prev_time`**：**上一次调度发起的时间点，初始化为0。**我们知道每执行1次推理阶段前，调度器都要做一次调度，这个变量存放的就是上次调度发起的时间点。
> - **`self.prev_prompt`**：取值为True/False，初始化为False。**若上一次调度时，调度器有从waiting队列中取出seq_group做推理，即为True，否则为False。**
> - **`self.last_prompt_latency`**：**记录“当前调度时刻（now） - 最后一次有从waiting队列中取数做推理的那个调度时刻”的差值**（并不是每一次调度时，调度器一定都会从waiting队列中取seq_group，它可能依旧继续对running队列中的数据做推理），初始化为0。

> - **`BlockManager`**：**物理块管理器**。这也是vLLM自定义的一个class。截止本文写作时，vLLM提供了`BlockSpaceManagerV1`和`BlockSpaceManagerV2`两个版本的块管理器。V1是vLLM默认的版本，V2是改进版本（但还没开发完，例如不支持prefix caching等功能）。所以本文依然基于`BlockSpaceManagerV1`进行讲解。物理块管理器这个class下又维护着两个重要属性：
>
> - - **`BlockAllocator`：物理块分配者，负责实际为seq做物理块的分配、释放、拷贝等操作。**这也是我们后文要解读的对象。其下又分成`self.gpu_allocator`和`self.cpu_allocator`两种类型，分别管理gpu和cpu上的物理块。
>   - **`self.block_tables`：负责维护每个seq下的物理块列表，本质上它是一个字典，形式如`{seq_id: List[PhysicalTokenBlock]}`。**注意，这里维护者【所有】seq_group下seq的物理块，而不是单独某一个seq的。因为整个调度器都是全局的，其下的BlockManager自然也是全局的。
>
> - **BlockManager只负责管理和分配物理块，映射关系潜藏在seq中**。理解这点对理解代码非常重要。

> - **如果当前swapped队列为空，那就去检查是否能从waiting队列中调度seq_group，直到不满足调度条件为止（gpu空间不足，或waiting队列已为空等）**。**此时，1个推理阶段中，所有的seq_group都处在prefill阶段。**
>
> - **如果当前swapped队列非空，或者无法从waiting队列中调度任何seq_group时：**
>
> - - 检查是否能从running队列中调度seq_group，直到不满足调度条件为止。
>   - 若本次无新的被抢占的seq_group，且swapped队列非空，就检查是否能从swapped队列中调度seq_group，直到不满足调度条件为止。
>
> **此时，1个推理阶段中，所有的seq_group要么全来自running队列，要么来自running + swapped队列，它们都处在decode阶段。**
>
> **至此我们要记住vLLM调度中非常重要的一点：在1个推理阶段中，所有的seq_group要么全部处在prefill阶段。要么全部处在decode阶段。**
>
> 你可能想问：**为什么要以swapped是否非空为判断入口呢？**
> 这是因为，如果当前调度步骤中swapped队列非空，说明在之前的调度步骤中这些可怜的seq_group因为资源不足被抢占，而停滞了推理。所以**根据FCFS规则，当gpu上有充足资源时，我们应该先考虑它们，而不是考虑waiting队列中新来的那些seq_group。**
> 同理，在图中你会发现，当我们进入对running队列的调度时（图中红色分支），我们会根据“**本次调度是否有新的被抢占的seq_group**”，来决定要不要调度swapped队列中的数据。这个理由也很简单：在本次调度中，我就是因为考虑到gpu空间不足的风险，我才新抢占了一批序列。既然存在这个风险，我就最好不要再去已有的swapped队列中继续调度seq_group了。

## _passed_delay()

判断调度waiting队列的时间点

> - **调度间隔设置得太小**，每次调度都只关心waiting中的新请求，这样发送旧请求的用户就迟迟得不到反馈结果。且此时waiting队列中积累的新请求数量可能比较少，不利于做batching，浪费了并发处理的能力。
> - **调度间隔设置得太大**，waiting中的请求持续挤压，同样对vLLM推理的整体吞吐有影响。

## can_allocate()：可以给prefill分配物理块

确实是否可以给这个seq_group分配物理块，做prefill

返回结果有三种情况：

* AllocStatus.NEVER：不分配；

* locStatus.OK：可以分配；

* AllocStatus.LATER：延迟分配

所以，假如我们想分配一个物理块

1. 先取出其waiting序列
2. 再选出其逻辑块
3. 再选出可用的物理块数量

> - **`self.watermark_blocks`：水位线block数量，它起的是一个预警和缓冲的作用**，防止在1次调度中把gpu上预留给KV Cache的显存空间打得过满，出现一些意外风险（毕竟这个预留的显存空间也是我们估计出来的）。
> - **NEVER和LATER的区别**：**这两者的相同之处在于，都是因为当前显存空间不够，而无法继续调度seq_group**。区别在于，**NEVER是因为这条seq实在太长（即prompt太长），长到动用了gpu上所有的block（num_total_gpu_blocks）都无法处理它**，所以后续步骤中我们会直接把这个seq标记为完成，不再处理它；而**LATER是因为之前可能已经调度了很多seq_group，它们占据了相当一部分显存空间，导致gpu上剩余的可用block（num_free_gpu_blocks）无法再处理它**，所以我们延迟处理。

## can_append_slot：可以给decode分配物理块

> **但这时，我们的物理块空间是用来做decode的（给每个seq分配1个token的位置），而不是用来做prefill的（给每个seq分配若干个token的位置），所以这里我们采取的是另一种判断方法`can_append_slot`。**
>
>
> 更具体来说，running队列中seq_group下的n个seqs在上1个推理阶段共生成了n个token。在本次调度中，我们要先为这n个token分配物理块空间，用于存放它们在本次调度中即将产生的KV值。
>
> 我们再回到这个seq_group的n个seqs上来，我们知道：
>
> - 当往1个seq的物理块上添加1个token时，可能有两种情况：
>
> - - 之前的物理块满了，所以我新开1个物理块给它
>   - 之前的物理块没满，我直接添加在最后一个物理块的空槽位上
>   - **所以，对于1个seq来说，最坏的情况就是添加1个物理块；对于n个seqs来说，最坏的情况就是添加n个物理块（想想原理篇中讲过的copy-on-write机制）**
>
> - **对于1个seq_group，除了那些标记为“finish”的seq外，其余seqs要么一起送去推理，要么一起不送去推理。即它们是集体行动的**
>
>
> **所以，判断能否对一个正在running的seq_group继续做推理的最保守的方式，就是判断当前可用的物理块数量是否至少为n。**

## allocate：给prefill分配物理块

涉及的细节太多（不同的prefix caching方式，逻辑块到物理块的映射，物理块释放，物理块的refcount即copy-on-write机制等等）

## append_slot：给decode分配物理块

涉及的细节太多（不同的prefix caching方式，逻辑块到物理块的映射，物理块释放，物理块的refcount即copy-on-write机制等等）

## preempt：抢占策略

> **在若干个推理阶段后，gpu上的资源不够了，这个seq_group不幸被调度器抢占（preemption）**，它相关的KV block也被swap out到cpu上。此时所有seq的状态变为swapped。这里要注意，**当一个seq_group被抢占时，对它的处理有两种方式：**
>
> - **Swap：如果该seq_group剩余生命周期中并行运行的最大seq数量 > 1，此时会采取swap策略**，即把seq_group下【所有】seq的KV block从gpu上卸载到cpu上。（seq数量比较多，直接把算出的KV block抛弃，比较可惜）
> - **Recomputation：如果该seq_group剩余生命周期中并行运行的最大seq数量 = 1，此时会采取recomputation策略**，即把该seq_group相关的物理块都释放掉，然后将它重新放回waiting队列中(**放在最前面**)。等下次它被选中推理时，就是从prefill阶段开始重新推理了，因此被称为“重计算”。（seq数量少，重新计算KV block的成本不高）

## Swap

看block_manager_v1的swap out逻辑



## excute

llm_emgine到executor到worker。GPU的是Worker，CPU的是CPUWorker。

然后Wroker执行Swap Cache操作后，通过model_runner进行计算。

### 

## budget机制

* 定义了一个SchedulingBudget
  * 维护的变量
    * token_budget：最大支持的token数目
    * max_num_seqs：最大支持的seqs数目
    * _requeset_ids_num_batched_tokens：标记同一个req已被登记过
    * _requeset_ids_num_curr_seqs：标记同一个req已被登记过
    * _num_batched_tokens：目前的tokens数目
    * _num_curr_seqs：目前的seqs数目
  * 维护的函数
    * can_schedule
      * 保证新增tokens和seqs后，_num_batched_tokens不超过token_budget和 _num_curr_seqs不超过max_num_seqs
    * remaining_token_budget
      * 获取剩余的token budget空间
    * add_num_batched_tokens
      * 将某一req的batch token加入到当前token数目中
    * subtract_num_batched_tokens
      * 将某一req的token标记和token数目都去除
    * add_num_seqs
      * 将某一req的seq加入到当前seq数目中
    * subtract_num_seqs
      * 将某一req的seq标记和seq数目都去除

总而言之，budget负责的就是维护token和seq不超过限定最大值

---

具体调度逻辑中

* 在running阶段
  * 通过budget获取最大num_running_tokens数目
    * **注意，这里running_queue是先来先服务的！实现控制chunked长度后，在之前的sort部分有可能prefill在decode前！导致后面的decode无法继续。**
  * 如果没位置了，需要swap出去，则删除该req的batch token和seqs
  * 如果可以推理
    * 通过add batch tokens添加添加req的当前token，但同一个req不会重复添加
      * **那么，假如chunk prefill后呢，会调整回1吗？貌似这里有点问题**
    * 假如会使用chunked prefill，才考虑num seqs，添加req对应的seqs
      * **这里说之前添加过了**
* 在swapped阶段和prefill阶段
  * 通过budget获取最大num_new_tokens数目
  * 利用budget的can_schedule判断
  * 新增换入的req的seq和token到budget中

需要注意的是，这里是通过每次调度新开一个Budget来实现更新！



## Output机制

1. _schedule做的是，输出SchedulerOutputs

其中，调度结果包含

* scheduled_seq_groups（参与调度所有seq）
* num_prefill_groups（prefill的数量）
* 当前batch包含的token数目
* 逻辑block的变化（swap in，swap out和copy）
* num_lookahead_slots（to be continued）
* running队列的大小
* preempted的大小

---

2. schedule在获取schedule后，用SchedulerOutputs的结果生成seq_group_metadata_lsit

* 把真正运行出来的中间变量弄出来
* sample的用处

---