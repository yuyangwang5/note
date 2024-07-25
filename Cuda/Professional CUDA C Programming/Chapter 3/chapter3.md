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

## Understanding the Nature of Warp Execution

