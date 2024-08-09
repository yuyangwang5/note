# GraphReduce: Large-Scale Graph Analytics on Accelerator-Based HPC Systems

## GAS模型
* **Gather阶段**  
每个节点收集来自入边和源节点的信息，并聚合；
* **Applay阶段**
使用gather的结果更新节点自身的值
* **Scatter阶段**
每个节点更新出边的状态

到论文发布为止，已有两种方式用于实现GAS模型：以边为中心和以点为中心。如下图所示：
![GAS的不同实现](./pic/2015%20GraphReduce/0%20GAS的不同实现.png "GAS的不同实现")

## GraphReduce框架

### 第一阶段 图划分
此阶段会将图的节点划分P个不相交的子集，每个子集维护一个名叫shard的数据结构，shard存有该子集所有节点的出边和入边。

在划分的时候希望：(1) shard包含的边数量大致相等；(2) 希望至少有一个shard能被完全加载到GPU内存；(3) 对于入边，我们按照其dst排序，而出边按照src排序，希望增强内存访问的局部性。

### 第二阶段 把数据传到GPU
本文考虑单GPU。假设我们要执行以下操作：(1) MemcpyAsync H2D(stream id)；(2) GPU Computation Task(stream id)；(3)MemcpyAsync D2H(stream id)。
若每个流都拥有以上三个操作，使用多个流的话操作可以被并行。而在GraphReduce，可以通过不同的流同时处理多个子图。

GraphReduce在此阶段希望达到：
1. 饱和主机和GPU间PCI-e总线的数据带宽；
2. 希望不同子图的传输和计算可以实现最大并发；
3. 安排并排序不同CUDA流，以保持GPU硬件忙碌状态。


### 第三阶段 计算
![子图结构](./pic/2015%20GraphReduce/1%20子图结构.png "子图结构")

上面为子图及其shard的实现结构。

GraphReduce实现的为GAS模型的一个变种，有四个阶段：
1. **Gather_Map**  
对于每条边，获取src状态，并存储于edge_update_array中，对应以边为中心；
2. **Gather_Reduce**   
对于每个dst，将获取到的消息进行聚合，对应以点为中心；
3. **Apply**   
更新节点状态，对应以点为中心；
4. **Scatter**   
依据出边，将节点的新状态传播出去，对应以边为中心。

该框架对并行的实现有两点：1. 每个子图给定阶段的计算在GPU上是并行的；2. 不同子图的计算也可以并发（驻留于GPU的多个碎片可以同时处理）。



![更新](./pic/2015%20GraphReduce/2%20更新.png "更新")