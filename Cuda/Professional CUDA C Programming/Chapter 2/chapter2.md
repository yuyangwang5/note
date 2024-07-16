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

# Managing Memory
CUDA runtime提供了管理device内存的函数，它们和C语言相似，如表格所示：

|STANDARD C FUNCTIONS|CUDA C FUNCTIONS|
|:--:|:--:|
|malloc|cudaMalloc|
|memcpy|cudaMemcpy|
|memset|cudaMemset|
|free|cudaFree|

* `cudaError_t cudaMalloc (void** devPtr, size_t size)`









