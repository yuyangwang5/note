## Local Search of Communities in Large Graphs

本论文只关注无向无权图，没有自环和多重边。

### 符号定义表
![符号定义表](./local_search_pic/1%20notations.png "notations")

### 问题定义

__DEFINITION__ **1** (COMMUNITY GOODNESS，社区优良度)  
设$G(V,E)$为图，$H \subseteq V$，$G[H]$为H导出子图。$G[H]$的定义为：  
$\delta (G[H]) = min\{deg_{G[H]}(v)|v \in H\}$

**PROBLEM** **DEFINITION** **1** (CSM, community search with the maximality constraint) 设$G(V,E)$为图，对于任意$v_0 \in V$，找到对应的$H\subseteq V$使得: (1). $v_0 \in H$; (2) $G[H]$连通; (3)$\delta (G[H])$在所有H中最大。

对于G和查询节点v，设m*(G,v)为可能的最大社区优良度，H*(G,v)为有最大优良度的任意社区，有：  
$0\le m*(G,v) \le deg_G(v)$   

某些应用中，我们可能对$\delta G[H] \le k$的社区感兴趣。由此，有问题2：

**PROBLEM DEFINITION 2** (CST, community search with threshold constraint) 对于G(V,E)，查询节点$v_0 \in V$，和一个常数k，寻找$H \subseteq V$，使得： (1) $v_0 \in H$; (2)G[H]连通; (3) $delta(G[H]) \ge K$。我们把它标记为$CST(k)$。

CST符合的结果可能会非常多，呈指数数量级。
![例图](./local_search_pic/2%20example%20graph.png "example graph")
以上图为例，若设置k=1，则结果有$2^{|V|-1}$个结果。  
因此，会只关注为CST寻找一个解决方案。其为问题3：

**PROBLEM DEFINITION 3** (mCST) 对于G(V,E)，查询节点$v_0 \in V$，和一个常数k，寻找$H \subseteq V$，使得： (1) $v_0 \in H$; (2)G[H]连通; (3) $\delta(G[H]) \ge k$; (4) H的大小最小。

然而，mCST为NP完全问题。

**LEMMA 1** 对于一个给定的G，一个节点$v_0$，一个整数k，若存在一个团使得$v_0 \in C$且|C|=k+1，那么C是CST(k)的最小解。

**PROBLEM DEFINITION 4** (MCC) 对于给定的图G=(V,E)和$v_0 \in V$，寻找包含$v_0$的最大团。

**LEMMA 2**
MCC是NP完全问题。  
**PROOF** MCC可以被归约到Maximal Clique(MC)问题上，该问题本身为NP完全问题。对于任意的图G=(V,E)，我们往G中加上$v_0$并将$v_0$与G中所有点连接，以此构建一个新的图G'=(V', E')。如此，G'中的MCC就是G中的MC。

**THEOREM 1** mCST是NP完全问题。  
**PROOF** 我们将mCST归约到MCC。有G(V,E)，查询节点$v_0$。MCC的决策问题为：G中是否有一个大小不少于k的团，其包含$v_0$。对于mCST，我们也可以构造决策问题：是否存在$H\subseteq V$，使得|H|=k，且符合mCST的限定条件：(1) $v_0 \in H$; (2)G[H]连通; (3) $\delta(G[H]) \ge k-1$。若H有解，则G[H]也有解。  
(疑惑：限制住|H|=k的证明是否严谨？这真的是mCST的决策问题吗？不应该对于选定的k，对|H|的各种大小进行决策判断吗？)

对于CSM和CST的关系，CST为CSM的决策问题。现在来建立CSM和CST之间的一些定量相关性，为之后寻找CST的解作准备：

**PROPOSITION 1** (DOWNWARD CLOSURENESS OF CST(k)，CST(k)的向下封闭性) 如果H是CST(k)的解决方案，那么对于任何$k_0 \le k$，H也是$CST(k_0)$的解决方案。

**PROPOSITION 2**. 给定G(V, E) 和一个查询顶点v，若H是CST(k)的解决方案，那么m∗(G, v)不小于k。

**PROPOSITION 3** (PRUNING RULE，修剪规则) 对于节点v，若$deg_G(v) \le k$，那么v不属于任何CST(k)的结果中。

借用命题1和命题2，对于查询节点v，我们可以在$[0, deg_G(v)]$这段区间利用二分查找，利用CST求解CSM。

在此声明，对于CSM和CST，均只寻找一个解决方案。

### 全局搜素
**DEFINITION 2** k-core，在本文中，除了满足一般k-core条件，还需要该图为最大子图。

**DEFINITION 3** (MAXIMUM CORE) 对于给定节点v，它最大核maxcore表示表示包含该点v的所有子图中的最大k-core值。

全局搜索算法见The Community-search Problem and How to Plan a Successful Cocktail Party 2010。

要利用全局算法解决CST(k)问题，假设v存在于CST(k)中某一个解决方案，那么v则为一个可接受的顶点。设A为CST(k)可接受顶点的集合。类似地，我们可以定义CSM问题可接受集合$A_0$。

**LEMMA 3** 对于G核一个查询节点v，若包含v的k-core连通分量$C_k$为最大连通分量，则其为CST(k)的解决方案。而对于其它解决方案H，有$H \subset C_k$。

**LEMMA 4** 对于G核一个查询节点v，若包含v的连通分量$C_{max}(v_0)$为CSM的解中最大连通分量，则而对于其它解决方案H，有$H \subset C_{max}(v_0)$。

**PROOF** 假设存在解H且其不属于C，即$H-C \neq \emptyset$，由于H和C中均有v，所以H和C连通。那么对于$H \cup C$，这将是更大的连通分量，且符合问题的解，这与C为最大连通分量这一前提条件不符合。

以上两个引理表明，如果v是CST(K)的可接受顶点，那么它必须要属于$C_k$。对于CSM有相似的结论。不幸的是，要获取所有可接受点，这一过程不亚于全局搜索（的开销）。

为了解决CST，只需迭代移除度数小于k的顶点即可；
为了解决CSM，设$G_0 = G$，之后不断移除最小顶点以及与之相连的边，获得$G_1, G_2, ..., G_t$，直到下一步中查询顶点v要被删除。之后，从$G_0 - G_t$中选出k最大的最大连通分量。
以上两种方法的时间复杂度均为$O(|V|+|E|)$。

### CST的局部搜索


