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








