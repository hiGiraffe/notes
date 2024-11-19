# Nsight System and Nsight Compute

Nsys用法：

```
"""
profile
"""
nsys profile –t cuda,osrt,nvtx,cpu –o baseline –w true python ....
nsys profile --stats=true python ...

"""
env
使用root用户可查看CPU信息
"""
nsys status --env
```

- [User Guide — nsight-systems 2024.6 documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

- [使用Nsight工具分析优化应用程序](https://cloud.baidu.com/doc/GPU/s/el8mizux4)

- [一文读懂nsight system与cuda kernel的时间线分析与可视化 - 知乎](https://zhuanlan.zhihu.com/p/691307737)

---

NCU用法：

- [NVIDIA性能分析工具nsight-compute入门 - 知乎](https://zhuanlan.zhihu.com/p/662012270)
- [Nsight 计算分析指南内核分析指南 - 吴建明wujianming - 博客园](https://www.cnblogs.com/wujianming-110117/p/17725564.html) 
- [nsight compute和nsight system的使用\_ncu --metrics-CSDN博客](https://blog.csdn.net/weixin_43838785/article/details/122128452)这里有各种参数

- [Achieved Occupancy](https://docs.nvidia.com/nsight-visual-studio-edition/4.6/Content/Analysis/Report/CudaExperiments/KernelLevel/AchievedOccupancy.htm) cuda的那些信息是什么
- [Memory Bound、Compute Bound 和 Latency Bound](https://zhuanlan.zhihu.com/p/673957960)
- [Nsight Compute快速上手指南（中文） - 夢番地](https://www.androsheep.win/post/ncu/)
- [Why Occupancy of GEMM is 12.5% - CUDA / CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/why-occupancy-of-gemm-is-12-5/237605/3)

---

sudo环境下如何使用conda 内的NCU

[sudo 权限下依然使用新建的anaconda环境](https://blog.csdn.net/u014447845/article/details/106780079)

```
sudo $(which ncu)

sudo $(which ncu) --set full -s 2000 -o decode -f python offline_inference.py
```





