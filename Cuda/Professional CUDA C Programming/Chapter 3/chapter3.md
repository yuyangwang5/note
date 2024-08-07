# CUDA EXECUTION MODEL
## Introducing the CUDA Execution Model

一般而言，运行模型会提供在特定架构计算机上，指令如何执行操作视图。CUDA运行模型提供了有关GPU并行架构的抽象视图，以帮助理解运行过程中线程的并发性（concurrency）。

### GPU Architecture Overview

GPU架构是围绕可扩展的流式多处理器（Streaming Multiprocessors, SM）阵列构建的。GPU硬件的并行性就依靠于GPU中许许多多的SM。

![Fermi_SM](./pic/1%20Fermi%20SM.png "Fermi_SM")

在上图中，展现了Fermi中的SM架构，从中可以看到一些关键的部件：
* CUDA核
* 共享内存/L1 Cache
* 寄存器文件
* 加载/存储单元
* 特殊函数单元
* Warp调度器

GPU中有许多的SM，每个SM支持上百个线程的并发执行。当一个内核网格被启动，网格中分布的线程块会被分派到可用的SM上执行。线程块被分配到SM伤后，线程块上的线程只会在该SM上执行。许多个线程块会被分配到同一个SM，其调度过程依照于SM上的可用资源。在单个线程内，会使用流水线以利用指令级并行。

CUDA采用SIMT(Single Instruction Multiple Thread)架构，以管理执行warp（32个线程组成的线程组）。warp中的线程会执行相同的指令，每个线程都有自己的指令地址计数器和寄存器状态，并在自己分配的数据上执行指令。每个SM会将线程块划分成多个warp，然后进行调度，使warp在可用硬件资源上执行。

SIMT能够使得同一个warp上的线程也能独立运行。与SIMD相比，SIMT的不同之处在于：
* 每一个线程有自己的指令地址计数器；
* 每个线程有自己的寄存器状态；
* 每个线程可以独立运行。

每个线程块只会被分配到一个SM上。一个线程块被分配到某个SM后，它会停留在该SM，直到执行结束。

逻辑视图和硬件视图上的对应关系如下图所示：

![软件与硬件的对应](pic/2%20Software%20and%20Hardware%20Correspondence.png "Software and Hardware Correspondence")

共享内存和寄存器在SM中是珍贵的资源。共享内存会被划分给驻存于SM的线程块，寄存器会被划分给线程。通过这些资源，同一线程块中的线程可以进行协作和交流。并不是所有的线程都会并行执行，同一线程块的不同线程可能在不同时间有不同的程序推进。

在并行的线程中共享数据会导致竞争：多个线程会以任意顺序访问同一数据，这将使结果难以预测。CUDA提供了使块中线程同步的方法，该方法使得所有线程到达程序某一点会停止执行，等待所有线程都执行到这一点，之后再继续执行。

在每个SM中，活跃warp的数目由SM的资源决定。当某个warp由于某些原因（例如，等待数据读取）而空闲，SM可以从任意（分配到该SM的）线程块中调用其它warp。SM的硬件资源被划分给了所有在SM上的线程、线程块，所有新调度的warp的状态已经存储于SM上，由此在并发的warp上切换并没有开销。

在SM上，寄存器和共享内存是稀缺的资源。CUDA将这些资源分配给了所有在该SM上的线程，由此将限制SM上同时活跃的warp的数目。同时活跃的warp的数目也就对应SM上的并行程度。

### Fermi架构

下图展示Fermi架构：
![Fermi架构](pic/3%20Fermi架构.png "3 Fermi架构")

Fermi架构由512个加速核(accelerator cores)，这些核叫CUDA核。每个CUDA核都有一个完全流水线化的整数算术逻辑单元（ALU）和一个浮点单元（FPU），每个时钟周期执行一个整数或浮点指令。Fermi架构上有16个SM，每个SM中有32个CUDA核。Fermi有6个384位GDDR5 DRAM内存接口，它能提供6GB的全局板载内存。host接口（host interface）通过PCI Express总线和CPU相连。图中Giga Thread是一个全局调度器，用于将线程块分配给各个SM的warp调度器。

Fermi有连续的768KB的L2缓存，它被16个SM共享。在Fermi架构中，每个SM包含：
* 执行单元（CUDA核）
* 调度和分派单元，用于调度warp
* 共享内存、寄存器文件、L1缓存

每一个SM都有16个加载/存储指令单元，每个周期为16个线程计算源和目标的地址。特殊函数单元（Special function unit, SFU）用于执行固有指令（如sine，cosine，平方根等）。每个SFU在每个时钟周期可以执行一个固有指令。

每个SM都有两个warp调度器和两个命令分派单元。当一个线程块被分配到SM，其中所有的线程会被分为warp。两个warp调度器会选择2个warp，并为每个warp分别传输1条指令，指令会被传输给16个CUDA核、16个load/store单元或者4个特殊计算单元（取决于指令本身）。其大致过程如下图所示：
![Fermi warp调度](./pic/4%20Fermi%20warp调度.png "Fermi warp调度")

Fermi有一个关键特性，它有64KB的片上可配置内存，它会被分为共享内存和L1缓存。共享内存使得块内线程可以协作，促进了芯片上数据的广泛重用，并大大减少了芯片外的数据传输。CUDA提供了运行时API以调整shared memory和L1缓冲的大小。修改片上内存配置可以提高程序表现。

Fermi还支持并发内核执行：同一个应用程序上下文中，可以启动的多个内核，使其同时在同一GPU上运行。Fermi可最多支持16个内核同时运行。

### Kepler架构
Kepler架构拥有15个SM和6个64位内存控制器。在Kepler上有三个创新：
* SM增强
* 动态并行
* Hyper-Q

其架构如下图所示：
![Kepler架构](./pic/5%20Kepler架构.png "Kepler架构")

Kepler SM包含192个单精度CUDA核、64个双精度单元、32个特殊函数单元和32个加载/存储单元。每个SM包含4个调度器和8个分派器，由此在一个SM中，可以同时运行4个Warp。在Kepler K20X(计算能力3.5)架构中，每个SM可以接收64个warp，即可以同时存储2048个线程。K20X架构将寄存器文件的大小增加到了64K（Fermi只有32K）。K20X在共享内存和L1缓存之间可以进行更多的内存划分（这应该是可以使共享内存和L1之间的比例调整更加细致）。K20X如下图所示：
![Kepler SM](./pic/6%20Kepler%20SM.png "Kepler SM")


动态并行（Dynamic Parallelism）意思是GPU可以动态启动新的网格。若没有动态并行，要启动内核函数必须要通过host，而有了动态并行之后，GPU就可以嵌套执行内核了，这样就消除了与CPU之间的交流开销。该特色带来的裨益如下图所示：
![动态并行](./pic/7%20动态并行.png "动态并行")

Hyper-Q在CPU和GPU之间增加了更多的硬件连接，如下图所示。由此，CPU核就可以同时调用多个GPU任务，更多任务可以在GPU运行。这可以增加GPU利用率，减少CPU的等待时间。在Fermi中，GPU与CPU之间只有一个硬件工作队列（hardware work queue）来传输任务，由此可能导致单个任务阻塞所有的任务。而Kepler提供了32个硬件工作队列，增强了GPU的并行性。
![Hyper-Q](./pic/8%20Hyper-Q.png "Hyper-Q")

### profile驱动的优化（Profile-Driven Optimization）
profiling会对程序进行分析，它会分析：
* 时/空间复杂度
* 特殊指令的使用
* 函数调用的频率和时长  

profile驱动的优化对于CUDA编程很重要，原因如下：
* 最简单的kernal实现表现并不会很好，profile可以帮助找到代码瓶颈。
* CUDA会划分SM中的资源给线程块。这种划分导致某些资源限制了表现。profiling工具可以获取资源利用信息。
* CUDA提供了硬件架构的抽象，由此我们可以控制线程同步。profiling工具可以帮助我们测量（相关数据）、（对相关数据）可视化，帮助我们找到可优化点。

CUDA提供了两个profiling工具：nvvp，可视化profiler；和nvprof，命令行profiler

在CUDA profiling中，事件（event）是一个可计数活动，它对应于一个硬件技术器，且会在kernel执行时被搜集。一些概念：
* 多数计数器所记录的信息是针对单个SM，而非整个GPU。
* 若只运行一次，只会搜集到少数计数器。有些计数器是互斥的（可能指的是运行一次，得到某个计数器的同时，与之互斥的计数器得不到），由此要运行多次才可以得到所有相关计数器。
* 由于GPU执行时的不确定性（如线程块分配顺序、warp调用顺序），多次运行时计数器的值不会相同。

有三个常见的kernel性能表现限制：
* 内存带宽
* 计算资源
* 指令、内存延迟

## Understanding the Nature of Warp Execution

### Warps and Thread Blocks
Warp是SM最基础的执行单元。一旦有线程块被分配到某个SM，该线程块中的线程会被分为warps。每个warp有32个（标号）连续的线程组成，且线程块中的线程会以SIMT的方式执行，即所有线程都会执行相同的指令，不过需要注意的是每个线程不一定处理相同的数据，它们可以有自己的数据。

每个block可以有1维、2维或3维，但从硬件角度上看，所有线程都是一维的。  
对于一维block，在一个warp中，线程的threadIdx.x是连续的。而对于二维或三维，它们的逻辑布局可以被转为一维的物理布局。如，对于2D线程块，块中每个线程的ID可以为：
$$threadIdx.y * blockDim.x + threadIdx.x$$

3维可以为：

$$threadIdx.z * blockIdx.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x$$

对于每个block的warps数，可以表示为：

$$WarpsPerBlock = \frac{ThreadsPerBlock}{warpSize}$$

因此，在硬件层面上，对于每一个块，分配的warp数是一致的。warp不会被不同的block区分（即不会一部分warp属于block，另一部分属于另一个block）。如果块大小不能被warp的大小整除，那么在最后一个warp中，会有线程不被激活。如下图所示：
![block分为warp](./pic/9%20block分为warp.png "block分为warp")

在图中，block中有80个线程，它被分为3个warp，最后一个warp中会有部分线程失活。

### Warp分歧，Warp Divergence
GPU没有复杂的分支预测机制。在一个时钟周期中，warp中所有线程必须执行同一个指令。如此，在执行中，若warp中线程要执行不同的指令会带来问题，这被称为warp divergence（warp分歧）。在warp分歧中，warp会有序执行每一个分支，且在执行某个分支时，会让没有执行该分支的线程失活。warp分支会导致程序执行表现的退化。
![warp divergece](./pic/10%20warp%20divergece.png "warp divergece")

由此，我们需要仅可能避免相同warp中存在不用的执行路径。由于对于thread而言，warp的分配实际上是确定的，由此可以尽可能在划分数据，使得在相同warp上的线程有相同的控制流。

有以下两个kernel：
``` c
__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.f;
    } else {
        b = 200.f;
    }
    c[tid] = a+b;
}

__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.f;
    } else {
        b = 200.f;
    }
    c[tid] = a+b;
}
```

在文中，作者利用nvprof工具对两个kernel进行profile，获取branch_efficiency特性。关于branch_efficiency，其计算公式为：
$$ Branch Efficiency = 100 \times \frac{\# Branches - \# Divergent Branches}{\#Branches} $$
而对两个kernel进行profile，发现两者均没有warp分歧出现（Branch Efficiency均为100%）。这是因为CUDA编译器进行了优化，将分支指令换成了条件指令，使得程序不是跳转（如此，程序执行路径会有多条，导致控制流发散），而是“满足某种条件而执行”，使得所有线程执行相同的指令。这中优化适用于短小的条件代码段。

在分支预测中，每个线程都有一个变量（谓词变量，predicate variable），类型为bool，可以依据条件被设置为1或0。由此，两个分支的控制流都会被执行，但只有变量为1对应线程的指令会被执行。若为0，线程并不执行，但也不会被阻塞。只有当条件语句体的指令少于某个阈值时，编译器才会将分支指令换为谓词指令（predicated instructions）。对于长代码路径，还是会有warp分歧出现。

第三个kernel：
``` c
__global__ void mathKernel3(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred) {
        a = 100.f;
    } 
    if (!ipred) {
        b = 200.f;
    }
    c[tid] = a+b;
}
```
（据说这个程序直接暴露了分支预测，不太清楚原理）  

可以在编译时取消优化内核函数。这时书上结果为：
```
mathKernel1: Branch Efficiency 83.33% 
mathKernel2: Branch Efficiency 100.00%
mathKernel3: Branch Efficiency 71.43%
```
可以看见CUDA还是有实现少许优化，使得kernel1和kernel3的分支效率有50%以上。

### Resource Partition
warp的本地执行上下文主要包括以下资源：
* 程序计数器
* 寄存器
* 共享内存

在warp的整个生命周期，其执行上下文都在片上（on-chip）。每个SM都有32位的寄存器集，它们存储于寄存器文件上，寄存器文件会被线程划分。共享内存的大小固定，它会被线程块瓜分。可以同时存储于SM的线程块和warp数是受限的，决定因素有：寄存器数；SM上可用共享内存数；kernel所要求的资源数。

![寄存器资源](./pic/11%20寄存器资源.png "寄存器资源")
上图展示了寄存器资源被线程消耗的情况。线程所需要的寄存器越多，能在SM上存储的warp越少。

![共享内存资源](./pic/12%20共享内存资源.png "共享内存资源")
上图展示了共享内存资源被线程块消耗的情况。若线程块占用过多的共享内存，那么能同时放进SM的线程块就会变少。

如果没有足够的寄存器或共享内存可以达到一个block的要求，那么kernel就会启动失败。

当计算资源（如寄存器、共享内存）被分配到某个线程块时，该块就是活跃块（active block），而其中的线程就叫做活跃warp（active warp）。活跃warp可以被进一步分为：
* 选中warp，Selected warp
* 阻塞warp，Stalled warp
* 待选warp， Eligible warp

warp调度器每个周期会活跃的warp并将它们分派到执行单元。正在执行的warp为 选中warp，若warp已经可以执行但还没有执行，就是 待选warp，warp还未满足执行条件则为 阻塞warp。warp可以被执行当且仅当：
* 32的CUDA核可以用于执行；
* 指令所需的参数已经准备好（这个准备没有直观的认知）

除此还存在硬件上的一些限制。例如，对于Kepler，一个SM从启动到结束，活跃warp数不能超过64个。任何周期 选中warp 数不能多余4个。如果warp被阻塞，warp调度器会选取待选warp将其换走。

CUDA编程中，需要注意对计算资源的划分，这会限制活跃warp的数目。为了最大化GPU的利用率，我们需要最大化活跃warp的数目。

### Latency Hiding

## Exposing Parallelism



