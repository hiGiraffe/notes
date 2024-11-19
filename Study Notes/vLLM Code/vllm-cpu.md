# Detailed explanation of vllm cpu decoding

## Template写法

```cpp
template <typename T, T count, typename F,
typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F &&f) {
    unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}
```

```cpp
namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F &&f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; 
```

使用例子

```cpp
vec_op::unroll_loop<int, head_elem_num_per_partition>(
    [&](int head_elem_idx) {
        if (head_elem_idx % 2 == 0) {
            vec_op::prefetch(next_v_block_cache_ptr +
                BLOCK_SIZE * head_elem_idx);
        }// 数据预取
    });
```

`T` 被指定为 `int`，`count` 被指定为 `head_elem_num_per_partition`，`F` 被指定为一个 lambda 表达式，这个 lambda 表达式接受一个 `int` 类型的参数 `head_elem_idx`。

## constexpr

作用包括：

1. **编译时求值：** 对于常量表达式，编译器可以在编译时计算其值，而不是在运行时计算。这样可以提高程序的性能和效率。
2. **编译时函数调用：** 对于被声明为 `constexpr` 的函数，在编译时可以被调用，并且其结果会在编译时求值，而不是在运行时计算。这使得可以在编译时进行复杂的计算和优化。
3. **常量表达式：** 可以使用 `constexpr` 来声明常量表达式，这样可以在编译时将其求值为常量，并且可以在需要常量表达式的地方使用。

## &&

`&&` 是右值引用的语法标记。在这段代码中，`&&f` 表示将可调用对象 `f` 绑定到右值引用上。右值引用允许我们对临时对象或可以移动的对象进行引用，通常用于提高性能和避免不必要的内存复制。`&&f` 表示对 `f` 的右值引用，允许我们在**不需要复制参数**的情况下传递它，并且可以在调用过程中保持其值类别。

## std::is_invocable_v<F, T>

- `std::is_invocable_v` 是一个 C++17 中引入的类型特征，用于检查是否可以使用给定类型参数调用给定的可调用对象类型。
- `F` 是可调用对象的类型，`T` 是作为参数传递给该可调用对象的类型。
- `std::is_invocable_v<F, T>` 返回一个布尔值，指示是否可以使用类型 `T` 的参数调用类型为 `F` 的可调用对象。

## std::enable_if_t<...>

- `std::enable_if_t` 是一个模板元函数，用于根据给定的条件启用或禁用模板。
- 如果条件为真，则 `std::enable_if_t` 返回模板参数的类型；如果条件为假，则不提供任何成员。

## make_integer_sequence<T, count>

[C++雾中风景16:std::make_index_sequence, 来试一试新的黑魔法吧 - HappenLee - 博客园](https://www.cnblogs.com/happenlee/p/14219925.html)

## std::forward

[浅谈std::forward](https://zhuanlan.zhihu.com/p/92486757)
std::forward通常是用于完美转发的，它会将输入的参数原封不动地传递到下一个函数中，这个“原封不动”指的是，如果输入的参数是左值，那么传递给下一个函数的参数的也是左值；如果输入的参数是右值，那么传递给下一个函数的参数的也是右值。

## lambda函数表达式

[C++ 11 Lambda表达式 - 滴水瓦 - 博客园](https://www.cnblogs.com/DswCnblog/p/5629165.html)
表达式
[capture list] (params list) mutable exception-> return type { function body }

1. capture list：捕获外部变量列表
2. params list：形参列表
3. mutable指示符：用来说用是否可以修改捕获的变量
4. exception：异常设定
5. return type：返回类型
6. function body：函数体

此外，我们还可以省略其中的某些成分来声明“不完整”的Lambda表达式，常见的有以下几种：

| 序号 | 格式                                                        |
| ---- | ----------------------------------------------------------- |
| 1    | [capture list] (params list) -> return type {function body} |
| 2    | [capture list] (params list) {function body}                |
| 3    | [capture list] {function body}                              |

例子中

```cpp
[&](int head_elem_idx) {
        if (head_elem_idx % 2 == 0) {
            vec_op::prefetch(next_v_block_cache_ptr +
                BLOCK_SIZE * head_elem_idx);
        }// 数据预取
    }
```

## [&]

`[&]` 是 lambda 表达式的捕获列表，用于指定 lambda 表达式如何捕获外部变量。

## 元编程std::integral_constant

[【C++ 泛型编程 进阶篇】：用std::integral_constant和std::is_*系列深入理解模板元编程-CSDN博客](https://blog.csdn.net/qq_21438461/article/details/131179100)

## Code

```cpp
// Paged attention v1
namespace {
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl {
  static void
  call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
       const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
       const int num_kv_heads, const float scale,
       const int
           *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
       const int *__restrict__ context_lens, // [num_seqs]
       const int max_num_blocks_per_seq,
       const float *__restrict__ alibi_slopes, // [num_heads]
       const int q_stride, const int kv_block_stride, const int kv_head_stride,
       const int num_seqs, const int num_heads) {
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    static_assert(BLOCK_SIZE == 16);

    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    const int parallel_work_item_num = omp_get_max_threads();

    size_t logits_bytes =
        parallel_work_item_num * max_context_len_padded * sizeof(float);
    float *logits = (float *)std::aligned_alloc(
        64, logits_bytes); // Cacheline alignment for each context token.
                           // [parallel_work_item_num, max_context_len_padded]

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) { //先循环seq的序号
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) { //再循环head的序号
      //一个seq一个head
        int context_len = context_lens[seq_idx]; //seq的上下文长度
        const int *seq_block_table =
            block_tables + max_num_blocks_per_seq * seq_idx; //指向seq块表的指针
        const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE; //进一法获取上下文存在block中的指针
        const int64_t kv_head_idx = head_idx / num_queries_per_kv; //GQA的优化，每个组内共享的kv heads的个数
        const scalar_t *__restrict__ q_vec_ptr =
            q + seq_idx * q_stride + head_idx * HEAD_SIZE; //指向token的q的数据
        const int last_block_token_num =
            context_len - (block_num - 1) * BLOCK_SIZE; //获取最后一个上下文block中的token数量
        float *__restrict__ thread_block_logits =
            logits + omp_get_thread_num() * max_context_len_padded; //logits缓冲区的矩阵地址

        // Compute logits 一个seq里多个block
        for (int block_idx = 0; block_idx < block_num; ++block_idx) { //对于每一个block的循环
          const int64_t physical_block_idx = seq_block_table[block_idx]; //物理块地址
          const scalar_t *__restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride; //物理块的索引
          float *__restrict__ head_block_logits =
              thread_block_logits + block_idx * BLOCK_SIZE; //logits缓冲块地址

          reduceQKBlockKernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, x>::call(
              q_vec_ptr, k_block_cache_ptr, head_block_logits, scale,
              block_idx == block_num - 1 ? last_block_token_num : BLOCK_SIZE); //计算QK，最后一个块的block数目可能不满
        }

        // Compute softmax，这里修改了logit中的值
        if (alibi_slopes) {
          reduceSoftmaxAlibi(thread_block_logits, context_len,
                             block_num * BLOCK_SIZE, alibi_slopes[head_idx], 0,
                             context_len);
        } else {
          reduceSoftmax(thread_block_logits, context_len,
                        block_num * BLOCK_SIZE);
        }

        // Compute value
        constexpr int head_elem_num_per_partition = 16; //每个分区中元素的数量
        constexpr int head_partition_num =
            HEAD_SIZE / head_elem_num_per_partition; //一个头有多少个分区
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) { //遍历每一个分区 fixme:
          vec_op::FP32Vec16 accums[head_elem_num_per_partition]; //创建一个累加器的数组，累加结果
          scalar_t *__restrict__ out_ptr =
              out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
              head_part_idx * head_elem_num_per_partition; //输出的地址
          for (int block_idx = 0; block_idx < block_num; ++block_idx) { //对于每一块
            const int64_t physical_block_idx = seq_block_table[block_idx]; //获取物理地址
            const float *__restrict__ prob_vec_ptr =
                thread_block_logits + block_idx * BLOCK_SIZE; //概率向量
            const scalar_t *__restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride +
                BLOCK_SIZE * head_part_idx * head_elem_num_per_partition; //获取block的物理地址
            reduceValueBlock<scalar_t, HEAD_SIZE, BLOCK_SIZE,
                             head_elem_num_per_partition>(
                prob_vec_ptr, v_block_cache_ptr, accums); //计算Value

            if (block_idx != block_num - 1) {
              const int64_t next_physical_block_idx =
                  seq_block_table[block_idx + 1];
              const scalar_t *__restrict__ next_v_block_cache_ptr =
                  v_cache + next_physical_block_idx * kv_block_stride +
                  kv_head_idx * kv_head_stride +
                  BLOCK_SIZE * head_part_idx * head_elem_num_per_partition;
              vec_op::unroll_loop<int, head_elem_num_per_partition>(
                  [&](int head_elem_idx) {
                    if (head_elem_idx % 2 == 0) {
                      vec_op::prefetch(next_v_block_cache_ptr +
                                       BLOCK_SIZE * head_elem_idx);
                    }// 数据预取
                  });
            }
          }

          vec_op::unroll_loop<int, head_elem_num_per_partition>(
              [&](int head_elem_idx) {
                float value = accums[head_elem_idx].reduce_sum();//求和得出value结果
                vec_op::storeFP32(value, out_ptr + head_elem_idx); //存到数据中
              });
        }
      }
    }
    std::free(logits);
  }
};
```

## Operation Logic

