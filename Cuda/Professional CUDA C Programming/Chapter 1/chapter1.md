# 1 Heterogeneous Parallel Computing with CUDA

## 并行 Parallelism

### 1. 任务并行 Task parallelism 和 数据并行 Data 

任务并行：存在多个任务或函数，可以许多个一起独立地并行运行。任务分布关注于在多个核上布置函数。

数据并行：有许多数据项，他们可以被同时处理。数据并行关注于在多核上分布数据。

CUDA编程十分适合数据并行。数据并行将要处理的数据项映射到线程上

### 2. 块划分 block partitioning 和 循环划分 cyclic partitioning

块划分：数据元素被分块，块数等于线程数。每一个块被分配到一个线程，每个线程只处理一个块。

循环划分：数据元素被分块，块数多于线程数。设线程数为N，线程i处理完第j个数据块后，去选择第j+N个数据块进行处理。

程序性能受块大小影响。最优划分与计算机架构紧密相关。

![数据分块图片](./pic/1%20数据分块.png "block partition and cyclic partition")

## Computer Architecture

### 1. Flynn's Taxonomy

![Flynn's Taxonomy](./pic/2%20Flynn's%20taxonomy.png "Flynn's Taxonomy")

* SISD
    - 对应serial architecture。计算机中只有一个核，每一次只有一条指令流在运行，操作于一条数据流上。
* SIMD 
    * 对应parallel architecture。计算机内有多个核，所有核执行相同的指令流，每条指令流作用于不同的数据流。
    * 程序员写程序时不用考虑并行，并行加速由编译器实现。
* MISD 
    * 不常见
* MIMD
    * 有多个核，每个核独立运行指令，作用于不同数据流。大多数MIMD可由多个SIMD作为子部件构成。

在计算架构方面，许多架构在实现：1. 减少延迟；2. 增加带宽；3. 增加吞吐量。

* 延迟 Latency
    * 一个操作开始到结束所花费的时间，常以毫秒计量。
* 带宽 Bandwidth
    * 单位时间处理的数据量，常以MB/s、GB/s计量。
* 吞吐量 Throughput
    * 单位时间处理的指令数，常以gflops（十亿浮点操作/秒）计量。

依照内存的组织方式，计算机架构可以进一步被分为分布式内存的多节点结构 Multi-node with distributed memory、共享内存的多处理器架构 Multiprocessor with shared memory。

* 分布式内存的多节点结构 Multi-node with distributed memory
  * 在多节点系统，大规模的计算引擎由多个处理器组成，它们通过网络进行互联，每一个处理器均有自己的内存。

![分布式内存的多节点结构](pic/3%20Multi-node%20with%20distributed%20memory.png "Multi-node with distributed memory")

* 共享内存的多处理器架构 Multiprocessor with shared memory
  * 多处理器架构中，处理器会连接到同一个内存，或共享一个低延迟连接（如PCI-Express）。多处理器可以是只有一个芯片，芯片中有多个核，这称为多核（multicore），也可以是拥有多个芯片，每个芯片可能也为多核。

![共享内存的多处理器架构](pic/4%20Multiprocessor%20with%20shared%20memory.png "Multiprocessor with shared memory")

众核（many-core）用于形容比多核的核还要多的多核架构，常有几十个或几百个核。GPU是一个众核架构。

## 异构计算 HETEROGENEOUS COMPUTUNG
### 异构架构
GPUs必须通过PCIe总线和CPU一起使用，也因此CPU称为host，GPU称为device。

![异构架构](pic/5%20heterogeneous%20architecture.png "heterogeneous architecture")

异构应用由host code和device code组成，其分别运行于CPU、GPU。CPU上的代码不仅要管理环境、代码，还要在计算任务加载进device前管理device上的数据。

NVIDIA的GPU计算平台（computing platform）可以在以下四类GPU上运行：
1. Tegra，用于嵌入式设备和移动设备；
2. GeForce，进行图形计算；
3. Quadro，用于专业视觉计算；
4. Tesla，用于数据并行计算。

在Tesla家族，有几款GPU：
1. Fermi，2010年出产，是世界上第一个完成的GPU架构。
2. Kepler，Fermi之后推出的GOU计算架构，在2012年出产。

GPU有两个主要的特性用于描述其性能：CUDA核数（Number of CUDA cores）和显存大小（Memory size）；由此，对应有两个指标用来评价GPU的表现：峰值计算性能（Peak computational performance）和显存带宽（Memory bandwidth）。峰值计算性能由gflops或tflops表示，显存带宽用GB/s表示。

NVIDIA使用compute capability来描述Tesla产品家族的GPU版本，具有相同主要修订号（应该是小数点之前的数字）的设备具有相同的核心架构。  
|架构|主要版本号|
|:----:|:--------:|
|Kepler|3|   
|Fermi|2|
|Tesla|1|     

NVIDIA提出的第一种架构和系列名称Tesla相同。

## 异构计算的范式 Paradigm of Heterogeneous Computing
CPU对于处理动态工作流（有少量计算操作和无法预测的控制流）更有优势，GPU则更擅长只有简单控制流的计算任务。由此，为了使程序表现最好，需要将串行任务放在CPU上，而处理数据的并行任务放于GPU上。

## CUDA: 一个异步编程的平台
CUDA是通用并行计算平台/编程模型，它可以利用NVIDIA GPUs中的并行计算引擎去解决许多复杂的计算问题。

CUDA C是标准ANSI C的扩展。CUDA提供两种API等级，以操作GPU、管理线程：CUDA Driver API和CUDA Runtime API。  
Driver API是更接近底层的API，它相对难以编程，但能对GPU有更多的控制；Runtime API更高级，它是用Driver API实现的。所有Runtime API编写的函数都会被分成更基础的操作，传给Driver API。

![CUDA API](pic/6%20CUDA%20API.png "CUDA API")

在使用时，对于使用何种API，并不会带来运行速度上的显著差别。运行时间更取决于其它因素，如内核如何使用内存、线程如何组织等。

NVIDIA' CUDA nvcc编译器会将device代码与host代码分开。device代码由一个个并行函数组成，每个并行函数为一个kernel。device代码会被nvcc编译。  
nvcc编译器是在LLVM开源编译器基础上编写的.






