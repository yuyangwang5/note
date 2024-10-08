# X-Stream: Edge-centric Graph Processing using Streaming Partitions

## X-Stream处理模型

X-Stream提供了两个API以处理图计算：Edge-centric scatter以及Edge-centric gather。前者以边为输入，根据边的源节点的值计算是否需要更新目的节点，若需要则将边加入update集合；而后者以update集合为输入，结合目的节点的值进行计算。

在每次迭代中，首先会进行scatter，之后再进行gather。scatter会作用在所有的边上，而gather作用于update集合中。X-Stream是同步的计算框架。 

X-Stream使用流来实现上述功能。流传输中，将数据分割成小的数据块（如字节或字符），并按照顺序逐个传输。输入流实现方法，从流中读取一个item；而输出流拥有的方法则是将一个item加入流尾。在scatter阶段，会以边作为输入，读取边、源节点数据，以及将需要更新的边（一条边，若其目的节点需要更新，则这条边称为update）加入输出流尾部。gather阶段，会将所有update作为输入，它不会产生输出流，对于update，其对应的目的节点数据会被更新。

图不一定能够放入内存。对于能完全放于内存的图，Cache为快速存储(Fast Storage)，主存(memory)为慢速存储(Slow storage)；对于内存放不下的图，主存(memory)为快速存储(Fast Storage)，SSD或硬盘(disk)为慢速存储(Slow storage)。

之所以使用流，是因为对于边和update流，支持在慢速存储上顺序访问；而对于节点数据，需要随机访问，而大图的节点数据可能无法直接直接放入快速存储，由此我们需要进行分图。

### Streaming Partition

流分区由点集、边集和update集组成。点集各分区间无重叠，边集包含所有源节点在点集中的边，update集动态确定，包含目的节点在边集中的边。初始化时，每个分区的点集和边集会被确定，且整个计算周期不会再改变。update集在每次gather前都会被重新计算。

在scatter阶段时，会遍历所有的分区，之后再进入下一阶段；gather亦是如此。对于每个分区，scatter阶段会读取其节点集，流式读进其边集，并计算生成update集的输出流，该流会不断被加入$U_{out}$集合中；这些update集会被重新组织，使得对于每个分区，其update集的目的节点在对应的点集中，这个阶段在本文中被称作shuffle阶段。shuffle阶段以scatter阶段的输出为输入，然后组织移动每个update到对应的$U_{in}(p)$集合中（p代表的是分区p）。shuffle结束后就开始执行gather阶段，gather阶段计算每个分区时，会先读入节点集、流式读入update集，计算新的数据值。

对于分区数目的选择，一是要让分区的节点数据可以被读入快速存储，二是要尽量小，以最大化对慢速存储的顺序访问（分区多，不连续的数据也更多）。

## 外核(Out-of-core)流式引擎
对于每一个分区，我们会存三个硬盘文件：节点、边集和update集。在实际执行时，会把shuffle阶段并入scatter中，将scatter阶段的update加到buffer中，当buffer满了直接执行shuffle操作，并将它们放入对应的分区的update文件中。

我们需要buffer存储各个阶段的输入输出，而变长数组开销大，此处我们设计一种数据结构，叫做流缓存（stream buffer）。流缓存包含两个部分，一个数组是大数组，叫做chunk数组；另一个是索引数组，有k项，对应k个分区。

可以使用两个流缓存以完成shuffle功能。一个用于存储scatter的输出，即shuffle的输入；一个用于存储shuffle的结果。我们会先遍历一遍输入流缓冲，计算每个分区有多少update，填写输出流缓冲的索引数组；再将输入流缓冲的update复制到输出流缓冲中。

该引擎开始时，会使用内存，在内存中执行shuffle，将边集分为不同的流分区。之后，引擎会进入循环。对于计算，有一些优化：
1. 如果所有的节点数据可以放入内存，则直接放入，可以减少节点的磁盘读取和写回；
2. scatter阶段，如果所有分区的update可以放入一个流缓冲中，则不用再写回内存，其可以被gather阶段直接使用。

### 硬盘I/O
为了保证顺序访问，在一个流缓冲满后，会继续读取，读入另一个流缓冲。同样，在将一个流缓冲写入硬盘时，scatter会继续执行，写入其它流缓冲。由此可见，对于输入和输出，均需要额外的流缓冲。我们称之为预取(prefetch)。

除此，X-Stream也会利用RAID的特性。RAID（独立磁盘冗余阵列）是一种存储技术，其是一种将多个独立的磁盘组成一个大的磁盘系统的技术方案。在RAID中，将大数据分割成若干个小块（stripe），然后将这些块放到不同的磁盘上，多个磁盘可以并行工作，可以提高带宽。在本系统中，可以将文件分布于不同的硬盘上，除此也可以将input和output文件放于不同磁盘上。X-Stream使用专用的I/O线程进行异步I/O操作，并为每个磁盘生成一个线程，由此可以将边和更新放在不同的磁盘上，并行执行I/O操作。

对于分图，除了希望每个分区的点可以写入内存外，我们还引入下面的约束：  
设update在所有流分区中均匀分布，为了实现最大的I/O带宽，需要S字节。由此，那么每一个chunk数组至少需要S\*K个字节，其中K表示的是分区数目。在设计中，输入输出各两个流缓存（以支持预取）；除此，还需要一个分区用于shuffle；由此，总共需要5个流缓存。假设顶点数据占用的总空间为N字节，内存有M字节，则可得：$\frac{N}{K} + 5SK \le M$。

对于上面的不等式，当$K=\sqrt{\frac{N}{5S}}$时，左式达最小值$2\sqrt{5NS}$。当S=16MB时，若节点数据为1TB，则K为120时取得最小M为17GB。这忽略了索引数组的开销。

## 内存(In-memory)流式引擎
此引擎处理的是能将顶点、边、更新均放入内存的图。内存引擎选择流分区的数量时，会考虑CPU缓存，目标是能够把每个分区的顶点数据放入CPU缓存中。由于在计算中需要边、update等数据，我们需要确保在计算时，CPU缓存引入除顶点数据外的数据时，不会将顶点数据换掉，由此将顶点占用空间设定为顶点数据大小、边数据大小和update大小之和。然后将所有顶点的总占用空间除以可用的CPU缓存大小，从而得到流分区数目。

该引擎需要三个流缓冲，一个用于存边，一个保存update，一个用于shuffle。计算时，首先加载入边数据，并shuffle进行流分区的划分，之后逐一处理每一个流分区。scatter阶段生成update，update放于输出流缓冲；接着输出流缓冲的update会被shuffle整理，每个分区对应形成一个块，然后用于gather阶段。

### 并行scatter-gather

对于不同的流分区，流操作互相独立。要考虑并行化的scatter-gather操作，需要考虑每个流分区能够获取到的共享缓存(shared cache)。此处假设每个核(core)可以获得大小相同的共享缓存大小。

执行不同流分区的线程需要将它们update附加于同一个chunk数组中。要实现这点，首先每个线程先将数据写入私有缓冲区（大小为8KB），然后先于chunk数组的末尾以原子操作保留一定空间，然后将数据写入该空间中。

由于每个分区边数不同，并行执行流分区可能显著地导致工作负载不均衡。由此，X-stream中有工作窃取机制，允许线程互相窃取流分区。













