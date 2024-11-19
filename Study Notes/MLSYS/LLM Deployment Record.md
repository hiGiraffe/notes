# llama2部署记录

[Llama 2: open source, free for research and commercial use](https://llama.meta.com/llama2/)

[Llama 2 in Hugging Face](https://huggingface.co/meta-llama)

[Llama 2 in Github](https://github.com/meta-llama/llama)

配置好CUDA、Pytorch，下载模型数据

```
pip install -e .
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```


## VLLM部署记录

**报错**

```
OSError: /home/cjl/llama/llama-2-7b does not appear to have a file named config.json.
```

谷歌搜索到是因为weight需要是hf格式，需要利用transformer提供的convert_llama_weights_to_hf.py脚本将其变为hf格式。

参考[transformers llama2 hugging face文档](https://huggingface.co/docs/transformers/main/en/model_doc/llama2)

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

其中在本机上transformers位置

```
/home/cjl/anaconda3/envs/vllm/lib/python3.9/site-packages/transformers/
```

```
python /home/cjl/anaconda3/envs/vllm/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py  \
    --input_dir /home/cjl/llama/llama-2-7b --model_size 7B --output_dir /home/cjl/llama/llama-2-7b-hf
```

**仍然报错**

```
RuntimeError: [enforce fail at inline_container.cc:424] 
```

谷歌没有搜到答案，后经验证确定，是由于硬盘空间不够。。

清理空间后，成功解决。



创建vllm_demo.py文件并运行，一个简单的demo就实现了。

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model='/home/cjl/llama/llama-2-7b-hf', dtype='half') 

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### chat completion功能

```
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/home/cjl/llama/llama-2-7b-hf",
        "messages": [
            {"role": "system", "content": "You are an intelligent British female writer and translator who is good at writing science fiction using multiple languages. You won a Nobel price in literature five years ago."},
            {"role": "user", "content": "Please detailedly tell a story about an exciting aerospace expedition for a Chinese boy Lam and his German dog. They are sent to aerospace by mistake and strive to wait for rescue from motherland with no water and food supply for over a month. They are almost caught by aliens disguised as his mother. Moreover, please translate the above story to Chinese, German, French, Portuguese and Japanese respectively."}
        ], "temperature": 0
    }'
```



## Accelerate部署记录

[Accelerate in Hugging Face](https://huggingface.co/docs/accelerate/en/index)

[Accelerate单机多卡简单demo](https://cloud.tencent.com/developer/news/1257333)



第一次使用一些优化设置，但貌似硬件不太适配，所以第二次设置了基本什么优化都没有的情况。

```
 `Accelerate` version: 0.29.0.dev0
- Platform: Linux-5.14.0-362.13.1.el9_3.x86_64-x86_64-with-glibc2.34
- Python version: 3.10.14
- Numpy version: 1.26.4
- PyTorch version (GPU?): 2.2.1+cu121 (True)
- PyTorch XPU available: False
- PyTorch NPU available: False
- PyTorch MLU available: False
- System RAM: 187.06 GB
- GPU type: Tesla V100S-PCIE-32GB
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: no
        - use_cpu: False
        - debug: True
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - gpu_ids: all
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - enable_cpu_affinity: False
        - downcast_bf16: no
        - tpu_use_cluster: False
        - tpu_use_sudo: False
        - tpu_env: []
```

## demo 1 简单使用

```python
from accelerate import Accelerator

from accelerate.utils import gather_object

accelerator = Accelerator()

# each GPU creates a string

message=[ f"Hello this is GPU {accelerator.process_index}" ] 

# collect the messages from all GPUs

messages=gather_object(message)

# output the messages only on the main process with accelerator.print() 

accelerator.print(messages)
```

需要注意我们要采用accelerate运行，而不是python运行。

```
$ accelerate launch acc_demo_1.py 
['Hello this is GPU 0', 'Hello this is GPU 1']
```
## llm demo
```python

from accelerate import Accelerator

from accelerate.utils import gather_object

from transformers import AutoModelForCausalLM, AutoTokenizer

from statistics import mean

import torch, time, json

accelerator = Accelerator()

# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books

prompts_all=[

    "The King is dead. Long live the Queen.",

    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",

    "The story so far: in the beginning, the universe was created.",

    "It was a bright cold day in April, and the clocks were striking thirteen.",

    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",

    "The sweat wis lashing oafay Sick Boy; he wis trembling.",

    "124 was spiteful. Full of Baby's venom.",

    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",

    "I write this sitting in the kitchen sink.",

    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",

] * 10

# load a base model and tokenizer

model_path="/home/cjl/llama/llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(

    model_path,   

    device_map={"": accelerator.process_index},

    torch_dtype=torch.bfloat16,

)

tokenizer = AutoTokenizer.from_pretrained(model_path)   

# sync GPUs and start the timer

accelerator.wait_for_everyone()

start=time.time()

# divide the prompt list onto the available GPUs 

with accelerator.split_between_processes(prompts_all) as prompts:

    # store output of generations in dict

    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference, prompt by prompt

    for prompt in prompts:

        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")

        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

        # remove prompt from output 

        output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # store outputs and number of tokens in result{}

        results["outputs"].append( tokenizer.decode(output_tokenized) )

        results["num_tokens"] += len(output_tokenized)

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs

results_gathered=gather_object(results)

if accelerator.is_main_process:

    timediff=time.time()-start

    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")

    print(results)
```
```
tokens/sec: 59.0, time 168.46036076545715, total tokens 10000, total prompts 100
```

## llm 批处理 demo

```python

from accelerate import Accelerator

from accelerate.utils import gather_object

from transformers import AutoModelForCausalLM, AutoTokenizer

from statistics import mean

import torch, time, json

accelerator = Accelerator()

def write_pretty_json(file_path, data):

    import json

    with open(file_path, "w") as write_file:

        json.dump(data, write_file, indent=4)

# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books

prompts_all=[

    "The King is dead. Long live the Queen.",

    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",

    "The story so far: in the beginning, the universe was created.",

    "It was a bright cold day in April, and the clocks were striking thirteen.",

    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",

    "The sweat wis lashing oafay Sick Boy; he wis trembling.",

    "124 was spiteful. Full of Baby's venom.",

    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",

    "I write this sitting in the kitchen sink.",

    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",

] * 10

# load a base model and tokenizer

model_path="models/llama2-7b"

model = AutoModelForCausalLM.from_pretrained(

    model_path,   

    device_map={"": accelerator.process_index},

    torch_dtype=torch.bfloat16,

)

tokenizer = AutoTokenizer.from_pretrained(model_path)   

tokenizer.pad_token = tokenizer.eos_token

# batch, left pad (for inference), and tokenize

def prepare_prompts(prompts, tokenizer, batch_size=16):

    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)] 

    batches_tok=[]

    tokenizer.padding_side="left"     

    for prompt_batch in batches:

        batches_tok.append(

            tokenizer(

                prompt_batch, 

                return_tensors="pt", 

                padding='longest', 

                truncation=False, 

                pad_to_multiple_of=2,

                add_special_tokens=False).to("cuda") 

            )

    tokenizer.padding_side="right"

    return batches_tok

# sync GPUs and start the timer

accelerator.wait_for_everyone()   

start=time.time()

# divide the prompt list onto the available GPUs 

with accelerator.split_between_processes(prompts_all) as prompts:

    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference in batches

    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=16)

    for prompts_tokenized in prompt_batches:

        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=100)

        # remove prompt from gen. tokens

        outputs_tokenized=[ tok_out[len(tok_in):] 

            for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

        # count and decode gen. tokens 

        num_tokens=sum([ len(t) for t in outputs_tokenized ])

        outputs=tokenizer.batch_decode(outputs_tokenized)

        # store in results{} to be gathered by accelerate

        results["outputs"].extend(outputs)

        results["num_tokens"] += num_tokens

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs

results_gathered=gather_object(results)

if accelerator.is_main_process:

    timediff=time.time()-start

    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
```

```
tokens/sec: 113.0, time elapsed: 87.8477463722229, num_tokens 10000
```



[Accelerate Handling big models for inference](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference)

修改device_map实现layer分层

修改前

```python
model = AutoModelForCausalLM.from_pretrained(

    model_path,   

    device_map={"": accelerator.process_index},

    torch_dtype=torch.bfloat16,

)
```



```
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [01:08<00:00, 22.69s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [01:10<00:00, 23.42s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
tokens/sec: 25.0, time 38.98043131828308, total tokens 1000, total prompts 10
device_map is  {'': 0}
[{'outputs': ['\nThe King is dead. Long live the Queen.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\n', 'They were sent to the country to stay with their eccentric uncle, who lived in a large house that had been in his family for hundreds of years.\nTheir uncle was a very strange man. He was tall and thin and had a long, hooked nose. He was always dressed in a long black cloak, and he had a long, white beard. He was very fond of children, and he was always telling them stories about the witches who lived in the', 'The universe was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, "Let there be light," and there was light. And God saw the light, that it was good; and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.\nAnd God said, "Let there', '\nWinston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.\nThe hallway smelt of boiled cabbage and old rag mats. At one end of it a colored poster, too large for indoor display, had been tacked to the wall. It', '\nIt is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.\n"It is a truth universally acknowledged, that a single man in possession of a good'], 'num_tokens': 500}]
```



修改后

```
model = AutoModelForCausalLM.from_pretrained(

    model_path,   

    device_map="auto",

    torch_dtype=torch.bfloat16,

)
```



```
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.21s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.13s/it]
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
tokens/sec: 12.0, time 78.71099257469177, total tokens 1000, total prompts 10
device_map is  {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'lm_head': 1}
[{'outputs': ['\nThe King is dead. Long live the Queen.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\nThe King is dead. Long live the King.\n', 'They were sent to the country to stay with their eccentric uncle, who lived in a large house that had been in his family for hundreds of years.\nTheir uncle was a very strange man. He was tall and thin and had a long, hooked nose. He was always dressed in a long black cloak, and he had a long, white beard. He was very fond of children, and he was always telling them stories about the witches who lived in the', 'The universe was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, "Let there be light," and there was light. And God saw the light, that it was good; and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.\nAnd God said, "Let there', '\nWinston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.\nThe hallway smelt of boiled cabbage and old rag mats. At one end of it a colored poster, too large for indoor display, had been tacked to the wall. It', '\nIt is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.\n"It is a truth universally acknowledged, that a single man in possession of a good'], 'num_tokens': 500}]
```

## Accelerate’s internal mechanisms

[Accelerate’s internal mechanisms](https://huggingface.co/docs/accelerate/main/en/concept_guides/internal_mechanism)

在Pytorch中加载预训练模型时

> 1. Create the model with randomly initialized weights
> 2. Load the model weights (in a dictionary usually called a state dict) from the disk
> 3. Load those weights inside the model

## Comparing performance between different device setups

[Comparing performance between different device setups](https://huggingface.co/docs/accelerate/main/en/concept_guides/performance)

* Setting the right seeds
* Observed Batch Sizes
* Learning Rates





Accelerater类

> The provided code snippet is a part of the `Accelerator` class from the Hugging Face `accelerate` library. The `Accelerator` class is designed to simplify distributed training and mixed precision training in deep learning applications. It provides a wrapper around various components such as models, optimizers, data loaders, and schedulers, and offers methods to manage these components in a distributed environment.
>
> Here's a brief overview of some key components and functionalities within the provided code snippet:
>
> 1. **Initialization (`__init__` method):** The constructor of the `Accelerator` class takes several arguments related to distributed training, mixed precision, data loading, and logging. It initializes various plugins like DeepSpeed, FullyShardedDataParallel (FSDP), and MegatronLM if they are available and configured.
>
> 2. **Attributes:** The class exposes several attributes that provide information about the training setup, such as the device in use, the distributed training configuration, the local and global process indices, and the mixed precision mode.
>
> 3. **Prepare Methods:** The `prepare`, `prepare_model`, `prepare_data_loader`, `prepare_optimizer`, and `prepare_scheduler` methods are used to wrap the respective objects (model, data loader, optimizer, and scheduler) with the necessary components for distributed training and mixed precision. These methods ensure that the objects are correctly set up to be used in a distributed environment.
>
> 4. **Training Utilities:** Methods like `backward`, `unscale_gradients`, `clip_grad_norm_`, and `clip_grad_value_` are provided to handle gradient scaling, clipping, and reduction in a distributed and mixed-precision context.
>
> 5. **Data Handling:** The `gather`, `gather_for_metrics`, and `pad_across_processes` methods are useful for aggregating data across processes and handling unevenly divided datasets in a distributed setting.
>
> 6. **Checkpointing:** The `save_state` and `load_state` methods allow for saving and loading the state of the training, including models, optimizers, schedulers, and other custom objects. These methods can handle sharding large models and work with different serialization methods.
>
> 7. **Memory Management:** The `free_memory` and `clear` methods help in releasing references to internal objects and managing memory efficiently during training.
>
> 8. **Mixed Precision Training:** The `autocast` context manager is provided to enable automatic mixed-precision training within a block of code.
>
> 9. **Hooks and Callbacks:** The class supports registering hooks for pre-saving and pre-loading states, allowing for custom behavior during checkpointing.
>
> 10. **Device Verification:** The `verify_device_map` method checks if a model has been prepared with a device map that resembles `auto`, which is not compatible with distributed training.
>
> This class is a powerful tool for researchers and developers working with deep learning models, as it simplifies the process of adapting training code for distributed environments and mixed-precision training. It abstracts away much of the complexity involved in these processes, making it easier to focus on the model architecture and training logic.
