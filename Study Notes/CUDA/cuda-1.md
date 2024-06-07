# Kernels

* ``__global__``  execute the kernel

* ``<<...>>``  execute configuration syntax
* ``threadIdx`` built-in variable

example:

```c++
∕∕ Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
    ...
    ∕∕ Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

# Thread Hierarchy

* ``threadIdx``  one-dimensional, two-dimensional, or three-
dimensional unique index accessible within the kernel 

* ``dim3``

  ```c++
  dim3 threadsPerBlock(16, 16); // 定义一个二维线程块，包含16行和16列的线程
  dim3 gridDim(32, 32); // 定义一个二维线程网格，包含32行和32列的线程块
  ```

* ``blockDim`` the dimension of the thread block
```c++
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}
int main()
{
    ...
    ∕∕ Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
```c++
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    C[i][j] = A[i][j] + B[i][j];
}
int main()
{
    ...
    ∕∕ Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

The second one can be used for larger matrices and utilize more GPU resources.

A thread block may contain up to **1024** threads.

* ``__syncthreads()``  acts as a barrier at which all threads in the block must wait before any is allowed to proceed.

# Thread Block Clusters

Cluster Launch allows you to specify the grouping size of thread blocks at compile time rather than dynamically at runtime. This allows for more efficient thread organization and management.

A maximum of **8** thread blocks in a cluster is supported as a portable cluster size in CUDA.

* ``__cluster_dims__(X,Y,Z)``

* ``<<<...>>>``

```c++
∕∕ Kernel definition
∕∕ Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{
    
}
int main()
{
    float *input, *output;
    ∕∕ Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    
    ∕∕ The grid dimension is not affected by cluster launch, and is still enumerated
    ∕∕ using number of blocks.
    ∕∕ The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

 The kernel can also be launched using the CUDA kernel launch API cudaLaunchKernelEx.

```c++
∕∕ Kernel definition
∕∕ No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}
int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    ∕∕ Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        ∕∕ The grid dimension is not affected by cluster launch, and is still enumerated
        ∕∕ using number of blocks.
        ∕∕ The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; ∕∕ Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

It should be noted that gridDim needs to be an integer multiple of blockDim.

# Memory Hierarchy
