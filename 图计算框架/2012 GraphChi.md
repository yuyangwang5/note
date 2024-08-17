# GraphChi: Large-Scale Graph Computation on Just a PC

## Parallel Sliding Windows (PSW)

PSW处理图有三个步骤：
1. 把图从硬盘加载进内存；
2. 更新点和边；
3. 将更新写回硬盘。

### Loading the Graph
G的点集V被分成P个不相交的interval。每个区间都有一个shard存储入边，入边的dst为interval中的一员。shard中边按照src进行排列，不同interval之间shard的大小大致相同。  
设P为分图数目，需要设置合适的P，使得任意一个shard都能放在内存之中。

被加载入内存执行计算的interval叫执行interval（execution interval）。为了计算该interval，首先该interval对应的shard必须加载入内存（载入入边），称该shard为内存shard（memory shard）。除此，interval对应节点的出边也要放入内存，出边在其它interval的shard中，称其它interval的shard为滑动shard（sliding shard）。对于其它interval的shard，我们需要寻找src在interval中的部分。由于shard按src排列，所以在其它shard中，执行interval的出边是连续的一块。而且，当执行下一个interval，sliding shard对应的出边部分会“滑”到紧接着的另一部分连续块，就像变长滑动窗口。由此，对于每个执行interval，PSW会执行P次硬盘读。

![PSW example](./pic/2012%20GraphChi/0%20PSW%20example.png "PSW example")

### Parallel Updates

在interval p载入内存后，PSW会在每一个点上并行执行用户定义的更新函数。若边的两个节点都在该interval中，可能会有竞争，由此我们需要增加外部约束：边的两点在同一个interval，会被标记为critical，两点间会增加顺序约束。对于非critical节点，它们之间可以完全并行。由此也算是引入了计算上的异步模型。

### Updating Graph to Disk
对于改变的边值，可以直接写于导入内存的边集中，它们可以看作是对磁盘边集的指针，若边集值有修改，其对应块也会被PSW修改。对于memory shard，它会被完全重写；对于sliding shard，它只有滑动窗口中的部分会被修改。当PSW的窗口滑动，其载入的value为新的value，此操作也是异步性的。

## Evolving Graphs 

对PSW进一步改进，使其能够支持增边操作。  

对于每个shard，我们可以将其分为P部分，第i部分边集的src在interval i中。对于第i个shard的第j部分，我们将其与内存中的edge-buffer(p, j)进行对应，其用户缓存增加的边。增边时，该边首先会被添加进该缓存区域中。当对应的interval加载入内存时，边缓存会算作内存图的一部分。

![边缓存](./pic/2012%20GraphChi/1%20边缓存.png "边缓存")

每一轮迭代，如果缓冲区中的边数超过一定数目，则会被写入硬盘。若shard最终变得过大，则需要将shard均分为两部分（或许此时图也该多一个interval？）

PSW也支持移边。移边操作后，对应边会被标记并忽视，当shard写回硬盘时会被永久删除。

若加边/移边会影响当前执行interval的节点，则加边/移边操作需要当前interval执行完才可以进行。

## 评价
### 局限性
* 无法动态安排子图计算顺序；
* 不适合遍历算法，因为难以取单个节点的邻居（需要完整的遍历）
### 技巧
* 可以让部分shard常驻内存，减少IO






