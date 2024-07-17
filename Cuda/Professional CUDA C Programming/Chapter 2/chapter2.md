# 2 CUDA Programming Model

## 引入CUDA编程模型 INTRODUCING THE CUDA PROGRAMMING MODEL

编程模型是对计算机架构的抽象，它是应用（Application）与底层硬件的桥梁。  
通信抽象是程序与编程模型实现之间的边界，它通过编译器或库使用特权硬件原语（privileged hardware primitives）和操作系统来实现。

![重要层抽象](pic/1%20重要层的抽象.png "the important layers of abstraction")

相比于一般的并行编程模型，CUDA编程模型提供了两个特殊功能：
* 组织线程
* 访问显存

从编程者视角，我们可以从不同层级去看待编程模型：
* Domain level
  * 当设计程序和算法时，我们关心的是如何分解数据和函数，这就是domain level；
* Logic level
  * 在编程阶段，我们关心如何组织线程，此为logic level；
* Hardware level
  * 需要理解线程如何映射到核上，这样可以帮助改进程序。

## CUDA Programming Structure
* Unified Memory
  * CUDA 6开始出现，它可以使编程人员通过单个指针访问CPU和GPU中的数据。

CUDA编程一个关键组成部分为内核（kernel），内核指的是在GPU上运行的代码。kernel可以看作一个串行程序，CUDA将会将kernel分配到线程上。  
当kernel运行时，控制权立马返回给host。

一个典型的CUDA程序处理流为：
1. 将CPU的数据复制到GPU；
2. 调用内核，在GPU存储的数据上进行计算；
3. 将数据从GPU复制回CPU。

## Managing Memory
CUDA runtime提供了管理device内存的函数，它们和C语言相似，如表格所示：

|STANDARD C FUNCTIONS|CUDA C FUNCTIONS|
|:--:|:--:|
|malloc|cudaMalloc|
|memcpy|cudaMemcpy|
|memset|cudaMemset|
|free|cudaFree|

* `cudaError_t cudaMalloc (void** devPtr, size_t size)`  
关于void**，首先，void\*为一个通用指针类型，可以用它指向GPU的地址。而通过void**，函数在GPU上开辟空间后，可以在存储void指针的地址，直接使void指针指向GPU中开辟出的区域。  
这个函数会在device内存上分配线性空间。

* `cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind)`  
kind可以有四种：1. cudaMemcpyHostToHost; 2. cudaMemcpyHostToDevice, 3. cudaMemcpyDeviceToHost; 4.cudaMemcpyDeviceToDevice。

以上这两个函数对于host都是同步函数，这些函数运行时，host会阻塞，等到它们返回结果后才会继续运行。实际上，除了kernel函数，其它所有CUDA函数都会返回枚举类型`cudaError_t`。如，若GPU的内存成功开辟，则返回：  
`cudaSuccess`  
否则会返回
`cudaErrorMemoryAllocation`  
若要将错误代码转化为可读信息，则可以使用函数：  
`char* cudaGetErrorString(cudaError_t error)`

在GPU的存储层次结构中，全局内存（global memory）和共享内存（shared memory）是两个很重要的存储类型。以CPU为参照，全局内存相当于CPU的内存层次，而共享内存相当于cache。GPU的共享内存可以由kernel直接控制。

![内存管理](pic/2%20内存管理.png "managing memory")

对于cuda内存控制函数应用示例：

``` c
float *d_A, *d_B, *d_C;
cudaMalloc((float**)&d_A, nBytes);
cudaMalloc((float**)&d_B, nBytes);
cudaMalloc((float**)&d_C, nBytes);
```
``` c
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
```
``` c
cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
```
``` c
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```
## Organizing Threads
由单个kernel启动所产生的所有线程统称为grid，grid里所有线程共享全局内存，一个grid由多个线程块组成；  
一个线程块（thread block）由一组线程组成，这些线程可以通过   
(1) 块内同步  
(2) 块内共享内存  
来进行协作。

Thread可以通过：  
(1) blockIdx（一个grid中对block的索引）  
(2) threadIdx（一个block中对thread的索引）  
来将自身与其它线程区别开来。这两个是内置且初始化好了的变量，可以被内核函数直接获取。  
这两个定位变量的类型为uint3，它是CUDA内置的向量类型，由最基本的整型派生而成。该结构包含3个无符号整型，分别对应x、y、z。即：  
```
  blockIdx.x
  blockIdx.y
  blockIdx.z

  threadIdx.x
  threadIdx.y
  threadIdx.z
```
grid和block三个维度的大小由两个变量指定：  
* blockDim
* gridDim
这两个变量的类型为dim3，它是基于uint3的整型向量类型，用于指定维度。在定义int3变量时，任何没有赋值的部分将会被初始化为1。dim3三部分依旧和x、y、z分别对应，如：
```
blockDim.x
blockDim.y
blockDim.z
```
下面程序示例kernel函数中如何获取线程的坐标。

``` c
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
    printf(
        "threadIdx: (%d, %d, %d)   blockIdx: (%d, %d, %d)   blockDim: (%d, %d, %d)   "
        "gridDim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, 
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, 
        gridDim.x, gridDim.y, gridDim.z
    );
}

int main() {
    int nElem = 6;

    dim3 block(3);
    dim3 grid ((nElem+block.x-1)/block.x);

    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);


    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

    return(0);
}
```
在用nvcc进行编译时，如果没有用-arch参数指定虚拟架构类型，那么默认虚拟架构为当前nvcc所支持的最低架构。



