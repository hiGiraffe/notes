# Brief Summary

* Functions which run on the host are prefaced with ``__host__`` in the function declaration. Kernels run on the device are prefaced with ``__global__``. Kernels that are run on the device and that are only called from the device are prefaced with ``__device__``

* The first step you should take in any CUDA program is to move the data from the host memory to device memory. The function calls ``cudaMalloc`` and `cudaMemcpy` allocate and copy data, respectively. `cudaMalloc` will allocate a specified number of bytes in the device main memory and return a pointer to the memory block, similar to malloc in C.

* The second step is to use `cudaMemcpy` from the CUDA API to transfer a block of memory from the host to the device. You can also use this function to copy memory from the device to the host. It takes four parameters, a pointer to the device memory, a pointer to the host memory, a size, and the direction to move data (`cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`). 

* Kernels are launched in CUDA using the syntax `kernelName<<<...>>>(...)`. The arguments inside of the chevrons (`<<<blocks, threads>>>`) specify the number of thread blocks and thread per block to be launched for the kernel. The arguments to the kernel are passed by value like in normal C/C++ functions.
* There are some read-only variables that all threads running on the device possess. The three most valuable to you for this assignment are `blockIdx`, `blockDim`, and `threadIdx`. Each of these variables contains fields x, y, and z. 
  * `blockIdx` contains the **x, y, and z coordinates** of the thread block where this thread is located. 
  * `blockDim` contains the **dimensions of thread block** where the thread resides. 
  * `threadIdx` contains the **indices** of this thread within the thread block.

# Code Details

## Initialize Context

```c++
cudaFree(0);
recurrence<<<1,1>>>(nullptr,nullptr,0,0); //recurrence here is a function name
```

1. Initialize cuda context to avoid including cost in timings later
2. Warm-up each of the kernels to avoid including overhead in timing.

## Allocate memory to the device arrays

```
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

- `devPtr`: A pointer to a pointer that will store the allocated device memory's address.
- `size`: The size of the memory to be allocated, specified in bytes.
- If the return value is `cudaSuccess`, the memory allocation was successful; otherwise, you may need to examine the specific error code to understand the reason for the allocation failure.

```
cudaError_t cudaFree(void* devPtr);
```

* `devPtr`: A pointer to the memory on the device that you want to free.
* If the return value is `cudaSuccess`, the memory deallocation was successful. Otherwise, you may need to examine the specific error code to understand the reason for the deallocation failure.

Example:

```C++
float *device_input_array = nullptr;
float *device_output_array = nullptr;

cudaMalloc(&device_input_array, num_bytes);
cudaMalloc(&device_output_array, num_bytes);

//....

cudaFree(device_input_array);
cudaFree(device_output_array);
```

## Implement the Kernel Function

Example:

```C++
void host_recurrence(vec &input_array, vec &output_array, size_t num_iter,
                     size_t array_size) {
  std::transform(input_array.begin(), input_array.begin() + array_size,
                 output_array.begin(), [&num_iter](elem_type &constant) {
                   elem_type z = 0;
                   for (size_t it = 0; it < num_iter; it++) {
                     z = z * z + constant;
                   }
                   return z;
                 });
}
```

to

```C++
__global__ void recurrence(const elem_type *input_array,
                           elem_type *output_array, size_t num_iter,
                           size_t array_length) {
  for (int xid = blockIdx.x * blockDim.x + threadIdx.x; xid < array_length;
       xid += blockDim.x * gridDim.x) {
    elem_type z = 0;
    elem_type constant = input_array[xid];
    int it = 0;
    while (it < num_iter) {
      z = z * z + constant;
      it++;
    }
    output_array[xid] = z;
  }
}
```

## Launch the Kernel

Example:

```C++
recurrence<<<grid_size, block_size>>>(d_input, d_output, num_iter,
                                        array_length);
```



## Calculate Time

```c++
#include <cuda_runtime.h>
```

```c++
typedef struct event_pair 
{
    cudaEvent_t start;
    cudaEvent_t end;
} event_pair;
```

```c++
inline void start_timer(event_pair *p) 
{
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}
```

```C++
inline double stop_timer(event_pair* p) 
{
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}
```

## GPU Error Checking

```C++
inline void check_launch(const char* kernel_name) 
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        return;

    std::cerr << "Error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "Error was: " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
}
```

