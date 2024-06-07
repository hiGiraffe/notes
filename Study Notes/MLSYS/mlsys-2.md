# 自动微分

* forward计算图

> ![image-20240417095542434](/images/dl-systems-2/1)

* backward计算图

> ![image-20240417095813942](/images/dl-systems-2/2)

* 同时需要考虑在不同道路中被使用的反向微分

> ![image-20240417100014694](/images/dl-systems-2/3)

* 反向自动微分代码

# 全连接

> A 𝐿-layer, fully connected network, a.k.a. **multi-layer perceptron (MLP)**, now with an explicit bias term, is defined by the iteration.
>
> ![image-20240417100826417](/images/dl-systems-2/4)
>
> 参数$\theta=\{W_{1:L},b_{1:L}\}$，$\sigma_{i}$一般是非线性的激活，一种常用的方法是$\sigma_{L}(x)=x$

# 优化器

* 梯度下降法

> ![image-20240417101443854](/images/dl-systems-2/5)
>
> 学习率$\times$梯度

* Newton’s Method

> 根据Hessian（二维导数矩阵）
>
> ![image-20240417102410999](/images/dl-systems-2/6)
>
> 等价于使用二阶泰勒展开将函数近似为二次函数，然后求解最优解

* Momentum

> 一种考虑更多的中间结构-momentum update，考虑先前梯度移动的平均值
>
> ![image-20240417103123998](/images/dl-systems-2/7)

* “Unbiasing” momentum terms
* Nesterov Momentum
* Adam

> Whether Adam is “good” optimizer is endlessly debated within deep learning, but it often seems to work quite well in practice (maybe?)

* Stochastic Gradient Descent

# Initialization

初始化跟大模型推理貌似无关，就没深入学习了

# Normalization 

需要看视频才看得懂，晚点补

# Regularization

需要看视频才看得懂，晚点补

# Transformer

![image-20240417105236671](/images/dl-systems-2/8)
