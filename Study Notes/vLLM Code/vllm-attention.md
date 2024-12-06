#  vLLM Paged Attention

## paged_attention_v1

* input
  * ``out: shape [num_seqs, num_heads, head_size]``
  * ``query: shape [num_seqs, num_heads, head_size]``
  * ``key_cache: shape [num_blocks, num_heads, head_size/x, block_size, x]``
  * ``value_cache: shape [num_blocks, num_heads, head_size, block_size]``
  * ``block_tables: shape [num_seqs, max_num_blocks_per_seq]``
  * ``num_kv_heads: num_heads``
  * ``context_lens: num_seqs``

x代表的是一个向量化的大小



* CUDA设置
  * ``gird: shape (num_heads, num_seqs, num_partition)`` 其中num_partition在不采用的时候为1
  * ``block: shape (NUM_THREADS)``





其中，对于attn_metadata，prefill的数据在前面，decode的数据在后面

```python
# NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|
    
num_prefills=num_prefills,
slot_mapping=slot_mapping_tensor, # token对应在table中的slot id
num_prefill_tokens=num_prefill_tokens, # prefill token的数目
num_decode_tokens=num_decode_tokens, # decode token的数目
seq_lens=seq_lens, # 各个句子的长度
seq_lens_tensor=seq_lens_tensor, # tensor类型的seq_lens，和上面没什么区别
max_query_len=max_query_len, # prefill阶段的query最大值，假如采用了chunk prefill，query_len，而不是context_len。比如484第一次chunked prefill算了20，则第二次max_query_len为464
max_prefill_seq_len=max_prefill_seq_len, # 可看上图
max_decode_seq_len=max_decode_seq_len, # Maximum sequence length among decode batch. 0 if there are prefill requests only.
query_start_loc=query_start_loc, # if the subquery length is [4, 6], it is [0, 4, 10]. 这个是query，假如decode，query是1
seq_start_loc=seq_start_loc, # if the sequence length is [4, 6], it is [0, 4, 10]. 这个是sequence
context_lens_tensor=context_lens_tensor, # context len的tensor，cache decode中不存储
block_tables=block_tables, # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
use_cuda_graph=use_captured_graph,
```

测试中的数据

```python
# 第一次 两个484传进去，prefill第一个484，第二个chunk28
num_prefills=2, num_prefill_tokens=512, num_decode_tokens=0, seq_lens=[28, 484], seq_lens_tensor=tensor([ 28, 484], device='cuda:0', dtype=torch.int32), max_query_len=484, max_prefill_seq_len=484, max_decode_seq_len=0, query_start_loc=tensor([  0,  28, 512], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0,  28, 512], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([0, 0], device='cuda:0', dtype=torch.int32), 
# 第二次，第一个推理485，第二个推理剩下的prefill
num_prefills=1, num_prefill_tokens=456, num_decode_tokens=1, seq_lens=[485, 484], seq_lens_tensor=tensor([485, 484], device='cuda:0', dtype=torch.int32), max_query_len=456, max_prefill_seq_len=484, max_decode_seq_len=485, query_start_loc=tensor([  0,   1, 457], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 485, 969], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([484,  28], device='cuda:0', dtype=torch.int32),
# 第三次 两个decode
num_prefills=0, num_prefill_tokens=0, num_decode_tokens=2, slot_mapping=tensor([2051253, 2050756], device='cuda:0'), seq_lens=[486, 485], seq_lens_tensor=tensor([486, 485], device='cuda:0', dtype=torch.int32), max_query_len=1, max_prefill_seq_len=0, max_decode_seq_len=486, query_start_loc=tensor([0, 1, 2], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 486, 971], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([485, 484], device='cuda:0', dtype=torch.int32)
```

如果对应一个剩下的prefill，block table照旧，slot mapping代表需要传进去的数据。



forward:

```

# decode部分
decode_query： [query decode的token数量, num_heads, head_size]
key_cache： [总的num_blocks, block_size, num_heads, head_size]
value_cache： [总的num_blocks, block_size, num_heads, head_size]
```



ops.paged_attention_v1参数

```
print(query.shape)
print(key_cache.shape)
print(value_cache.shape)
print(num_kv_heads)
print(block_tables.shape)
print(seq_lens)
print(block_size)
print(max_seq_len)
```

30b

```
torch.Size([1, 56, 128])
torch.Size([195, 56, 16, 16, 8]) 56是num_heads 16(2)和8(4)相乘是head_sizes 16(3)是block_sizes
torch.Size([195, 56, 128, 16])
56
torch.Size([1, 13])
tensor([204], dtype=torch.int32)
16
204
```


125m

```
torch.Size([1, 12, 64])
torch.Size([7281, 12, 8, 16, 8])
torch.Size([7281, 12, 64, 16])
12
torch.Size([1, 13])
tensor([204], dtype=torch.int32)
16
204
```



attn_backend_impl的数据

```
print(num_heads)
print(head_size)
print(scale)
print(num_kv_heads)
print(alibi_slopes)
print(sliding_window)
print(kv_cache_dtype)
print(blocksparse_params)
```

opt-125m

```
12
64
0.125
12
None
None
auto
None
```

opt-30b

```
56
128
0.08838834764831845
56
None
None
auto
None
```

---

torch_sdpa make_metadata数据

prefill [['0', 195], ['1', 195], ['2', 195], ['3', 195], ['4', 195]]时参数：

```
is_prompt: True
seq_lens: [195, 195, 195, 195, 195]
seq_lens_tensor: None
max_decode_seq_len: None
num_prefills: 5
num_prefill_tokens: 975
num_decode_tokens: 0
block_tables: tensor([])
slot_mapping: tensor([15984, 15985, 15986, 15987, 15988, 15989, 15990, 15991, 15992, 15993,
        15994, 15995, 15996, 15997, 15998, 15999, 15968, 15969, 15970, 15971,
        15972, 15973, 15974, 15975, 15976, 15977, 15978, 15979, 15980, 15981,
        15982, 15983, 15952, 15953, 15954, 15955, 15956, 15957, 15958, 15959,
        15960, 15961, 15962, 15963, 15964, 15965, 15966, 15967, 15936, 15937,
        15938, 15939, 15940, 15941, 15942, 15943, 15944, 15945, 15946, 15947,
        15948, 15949, 15950, 15951, 15920, 15921, 15922, 15923, 15924, 15925,
        15926, 15927, 15928, 15929, 15930, 15931, 15932, 15933, 15934, 15935,
        15904, 15905, 15906, 15907, 15908, 15909, 15910, 15911, 15912, 15913,
        15914, 15915, 15916, 15917, 15918, 15919, 15888, 15889, 15890, 15891,
        15892, 15893, 15894, 15895, 15896, 15897, 15898, 15899, 15900, 15901,
        15902, 15903, 15872, 15873, 15874, 15875, 15876, 15877, 15878, 15879,
        15880, 15881, 15882, 15883, 15884, 15885, 15886, 15887, 15856, 15857,
        15858, 15859, 15860, 15861, 15862, 15863, 15864, 15865, 15866, 15867,
        15868, 15869, 15870, 15871, 15840, 15841, 15842, 15843, 15844, 15845,
        15846, 15847, 15848, 15849, 15850, 15851, 15852, 15853, 15854, 15855,
        15824, 15825, 15826, 15827, 15828, 15829, 15830, 15831, 15832, 15833,
        15834, 15835, 15836, 15837, 15838, 15839, 15808, 15809, 15810, 15811,
        15812, 15813, 15814, 15815, 15816, 15817, 15818, 15819, 15820, 15821,
        15822, 15823, 15792, 15793, 15794, 15776, 15777, 15778, 15779, 15780,
        15781, 15782, 15783, 15784, 15785, 15786, 15787, 15788, 15789, 15790,
        15791, 15760, 15761, 15762, 15763, 15764, 15765, 15766, 15767, 15768,
        15769, 15770, 15771, 15772, 15773, 15774, 15775, 15744, 15745, 15746,
        15747, 15748, 15749, 15750, 15751, 15752, 15753, 15754, 15755, 15756,
        15757, 15758, 15759, 15728, 15729, 15730, 15731, 15732, 15733, 15734,
        15735, 15736, 15737, 15738, 15739, 15740, 15741, 15742, 15743, 15712,
        15713, 15714, 15715, 15716, 15717, 15718, 15719, 15720, 15721, 15722,
        15723, 15724, 15725, 15726, 15727, 15696, 15697, 15698, 15699, 15700,
        15701, 15702, 15703, 15704, 15705, 15706, 15707, 15708, 15709, 15710,
        15711, 15680, 15681, 15682, 15683, 15684, 15685, 15686, 15687, 15688,
        15689, 15690, 15691, 15692, 15693, 15694, 15695, 15664, 15665, 15666,
        15667, 15668, 15669, 15670, 15671, 15672, 15673, 15674, 15675, 15676,
        15677, 15678, 15679, 15648, 15649, 15650, 15651, 15652, 15653, 15654,
        15655, 15656, 15657, 15658, 15659, 15660, 15661, 15662, 15663, 15632,
        15633, 15634, 15635, 15636, 15637, 15638, 15639, 15640, 15641, 15642,
        15643, 15644, 15645, 15646, 15647, 15616, 15617, 15618, 15619, 15620,
        15621, 15622, 15623, 15624, 15625, 15626, 15627, 15628, 15629, 15630,
        15631, 15600, 15601, 15602, 15603, 15604, 15605, 15606, 15607, 15608,
        15609, 15610, 15611, 15612, 15613, 15614, 15615, 15584, 15585, 15586,
        15568, 15569, 15570, 15571, 15572, 15573, 15574, 15575, 15576, 15577,
        15578, 15579, 15580, 15581, 15582, 15583, 15552, 15553, 15554, 15555,
        15556, 15557, 15558, 15559, 15560, 15561, 15562, 15563, 15564, 15565,
        15566, 15567, 15536, 15537, 15538, 15539, 15540, 15541, 15542, 15543,
        15544, 15545, 15546, 15547, 15548, 15549, 15550, 15551, 15520, 15521,
        15522, 15523, 15524, 15525, 15526, 15527, 15528, 15529, 15530, 15531,
        15532, 15533, 15534, 15535, 15504, 15505, 15506, 15507, 15508, 15509,
        15510, 15511, 15512, 15513, 15514, 15515, 15516, 15517, 15518, 15519,
        15488, 15489, 15490, 15491, 15492, 15493, 15494, 15495, 15496, 15497,
        15498, 15499, 15500, 15501, 15502, 15503, 15472, 15473, 15474, 15475,
        15476, 15477, 15478, 15479, 15480, 15481, 15482, 15483, 15484, 15485,
        15486, 15487, 15456, 15457, 15458, 15459, 15460, 15461, 15462, 15463,
        15464, 15465, 15466, 15467, 15468, 15469, 15470, 15471, 15440, 15441,
        15442, 15443, 15444, 15445, 15446, 15447, 15448, 15449, 15450, 15451,
        15452, 15453, 15454, 15455, 15424, 15425, 15426, 15427, 15428, 15429,
        15430, 15431, 15432, 15433, 15434, 15435, 15436, 15437, 15438, 15439,
        15408, 15409, 15410, 15411, 15412, 15413, 15414, 15415, 15416, 15417,
        15418, 15419, 15420, 15421, 15422, 15423, 15392, 15393, 15394, 15395,
        15396, 15397, 15398, 15399, 15400, 15401, 15402, 15403, 15404, 15405,
        15406, 15407, 15376, 15377, 15378, 15360, 15361, 15362, 15363, 15364,
        15365, 15366, 15367, 15368, 15369, 15370, 15371, 15372, 15373, 15374,
        15375, 15344, 15345, 15346, 15347, 15348, 15349, 15350, 15351, 15352,
        15353, 15354, 15355, 15356, 15357, 15358, 15359, 15328, 15329, 15330,
        15331, 15332, 15333, 15334, 15335, 15336, 15337, 15338, 15339, 15340,
        15341, 15342, 15343, 15312, 15313, 15314, 15315, 15316, 15317, 15318,
        15319, 15320, 15321, 15322, 15323, 15324, 15325, 15326, 15327, 15296,
        15297, 15298, 15299, 15300, 15301, 15302, 15303, 15304, 15305, 15306,
        15307, 15308, 15309, 15310, 15311, 15280, 15281, 15282, 15283, 15284,
        15285, 15286, 15287, 15288, 15289, 15290, 15291, 15292, 15293, 15294,
        15295, 15264, 15265, 15266, 15267, 15268, 15269, 15270, 15271, 15272,
        15273, 15274, 15275, 15276, 15277, 15278, 15279, 15248, 15249, 15250,
        15251, 15252, 15253, 15254, 15255, 15256, 15257, 15258, 15259, 15260,
        15261, 15262, 15263, 15232, 15233, 15234, 15235, 15236, 15237, 15238,
        15239, 15240, 15241, 15242, 15243, 15244, 15245, 15246, 15247, 15216,
        15217, 15218, 15219, 15220, 15221, 15222, 15223, 15224, 15225, 15226,
        15227, 15228, 15229, 15230, 15231, 15200, 15201, 15202, 15203, 15204,
        15205, 15206, 15207, 15208, 15209, 15210, 15211, 15212, 15213, 15214,
        15215, 15184, 15185, 15186, 15187, 15188, 15189, 15190, 15191, 15192,
        15193, 15194, 15195, 15196, 15197, 15198, 15199, 15168, 15169, 15170,
        15152, 15153, 15154, 15155, 15156, 15157, 15158, 15159, 15160, 15161,
        15162, 15163, 15164, 15165, 15166, 15167, 15136, 15137, 15138, 15139,
        15140, 15141, 15142, 15143, 15144, 15145, 15146, 15147, 15148, 15149,
        15150, 15151, 15120, 15121, 15122, 15123, 15124, 15125, 15126, 15127,
        15128, 15129, 15130, 15131, 15132, 15133, 15134, 15135, 15104, 15105,
        15106, 15107, 15108, 15109, 15110, 15111, 15112, 15113, 15114, 15115,
        15116, 15117, 15118, 15119, 15088, 15089, 15090, 15091, 15092, 15093,
        15094, 15095, 15096, 15097, 15098, 15099, 15100, 15101, 15102, 15103,
        15072, 15073, 15074, 15075, 15076, 15077, 15078, 15079, 15080, 15081,
        15082, 15083, 15084, 15085, 15086, 15087, 15056, 15057, 15058, 15059,
        15060, 15061, 15062, 15063, 15064, 15065, 15066, 15067, 15068, 15069,
        15070, 15071, 15040, 15041, 15042, 15043, 15044, 15045, 15046, 15047,
        15048, 15049, 15050, 15051, 15052, 15053, 15054, 15055, 15024, 15025,
        15026, 15027, 15028, 15029, 15030, 15031, 15032, 15033, 15034, 15035,
        15036, 15037, 15038, 15039, 15008, 15009, 15010, 15011, 15012, 15013,
        15014, 15015, 15016, 15017, 15018, 15019, 15020, 15021, 15022, 15023,
        14992, 14993, 14994, 14995, 14996, 14997, 14998, 14999, 15000, 15001,
        15002, 15003, 15004, 15005, 15006, 15007, 14976, 14977, 14978, 14979,
        14980, 14981, 14982, 14983, 14984, 14985, 14986, 14987, 14988, 14989,
        14990, 14991, 14960, 14961, 14962])
```

decode [['0', 196], ['1', 196], ['2', 196], ['3', 196], ['4', 196]]时数据

```
is_prompt: False
slot_mapping: tensor([15795, 15587, 15379, 15171, 14963])
seq_lens: [196, 196, 196, 196, 196]
seq_lens_tensor: tensor([196, 196, 196, 196, 196], dtype=torch.int32)
max_decode_seq_len: 196
num_prefill_tokens: 0
num_decode_tokens: 5
num_prefills: 0
block_tables: tensor([[999, 998, 997, 996, 995, 994, 993, 992, 991, 990, 989, 988, 987],
        [986, 985, 984, 983, 982, 981, 980, 979, 978, 977, 976, 975, 974],
        [973, 972, 971, 970, 969, 968, 967, 966, 965, 964, 963, 962, 961],
        [960, 959, 958, 957, 956, 955, 954, 953, 952, 951, 950, 949, 948],
        [947, 946, 945, 944, 943, 942, 941, 940, 939, 938, 937, 936, 935]],
       dtype=torch.int32)
```

