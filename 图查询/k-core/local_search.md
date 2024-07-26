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

#### 简单算法

首先我们分析$\delta(\cdot)$的单调性。考虑从一个查询顶点v开始，不断添加节点，最终得到一个解H。假设节点序列为$v_0, v_1, ...,v_t$。由于$\delta(\cdot)$非单调，所以$\delta(H_i)$不一定小于$\delta(H_{i+1})$。不过，存在一种添加节点的顺序，使得$\delta(H_i)$关于i非递减。

**THEOREM 2** 对于$v_0 \in H$，存在某一节点序列$v_0, v_1, ..., v_t$，使得$\forall 0\le i \lt t, \delta(H_i) \le \delta(H_{i+1})$。

**PROOF** 这相当于证明我们可以从H中不断移除顶点，直到只剩下$v_0$，且每次顶点都不会增加剩余顶点的最小度数。假设当前集合是$H'$，如果$H' = \{ v_0 \}$，我们就找到了序列；如果$H' = \{ v_0, v_i \}$，移除v_i后，要么降低最小度，要么最小度不变。如果$H'$节点不止两个，这样情况下，必然存在一个节点v，有$\delta (G[H']) \ge delta (G[H'] - {v})$。如果这样的节点不存在，意为无论怎么移除顶点，最小度都会增加。若希望移除顶点后最小度增加，那么移除的顶点必须是图中度数最小的顶点。也就是说，该图中所有点的度数都相等。而在这种情况下，无论怎么移除顶点，都不会使得最小度增大。

由定理2，可以得到一种简单算法。
该算法流程如下图所示：
![Algorithm 1 search](./local_search_pic/3%20Algorithm%201%20search.png "Algorithm 1 search")

最开始，$H' = \{v_0\}$，然后调用search函数。search函数遍历H'邻居，若某个邻居使得$\delta(H' \cup \{v\}) \ge \delta(H')$，则将$H' \cup \{v\}$作为输入进行新一轮查找。这相当于深度搜索。一旦找到解，就返回。这种算法的复杂度是指数级别的。

####  解决CST问题的算法框架, A Framework for Solving CST

接下来要介绍另一个算法。该算法包含三步：
1. 通过存在CST(k)解的充分条件，判断图中是否可能存在解；
2. 使用函数candidateGeneration()，从查询节点v周围选取候选集C。在选取候选集时，若在某一步候选集符合条件，甚至可以直接返回答案；
3. 在候选集导出子图中进行全局搜索。

算法流程如下图所示：
![A General Framework of CST](./local_search_pic/4%20Algorithm%202.png "A General Framework of CST")

大部分情况下，到第二步就能找到合适的解。

只要在第2步，candidateGeneration()没有忽略掉任何可接受点，那么该算法一定能返回有效的解（否则无解）。

**PROPOSITION 4** 对于图G和查询节点v，若$H \in V$时CST(k)的一个答案，那么对于任意$H'\in V$，$G[H\cup H']$为CST(k)的解（两个不同解的并集也是解）。

首先，我们先关注算法的第一步，关于判断图的有效性。此处提出一种有效性判断，该判断可以给出$m*(G,v)$的上界：

**THEOREM 3** 设G为连通图，对于任意点v，有：
$$ m^*(G, v) \le \lfloor \frac{1 + \sqrt{9+8(|E|-|V|)}}{2} \rfloor $$

**PROOF** 由于G为连通图，有$|E| \ge |V| - 1$。使用H\*和m\*分别表示最优社区和对应的最小度。则G[H\*]至少拥有$\lceil m^*|H^*|/2 \rceil$条边。对于剩下的节点V-H*，它们之间有V-H*-1条边，而剩下节点与H*连通又需要至少一条边，则有：
$$\lceil\frac{ m^*|H^*|}{2} \rceil + |V| - |H^*| \le |E|$$
又因为有 $|H^*| \ge m^* + 1$，所以可进一步化为$(\frac{m^*}{2} - 1)(m^*+1) \le |E|-|V|$。经过整理就可得到定理3的公式。

现在开始研究第二步使用的函数candidateGeneration()。这里给出一种直接简单的实现。实际上，该算法就是利用BFS将v邻近区域中节点度数不小于k的点都加入候选集。伪代码如下所示：
![Naive candidateGeneration](./local_search_pic/5%20Algorithm%203.png "Naive candidateGeneration")

对于算法2，若其使用算法3为candidateGeneration函数，那么第三步全局搜索的复杂度和第二步相同。此时，算法2的时间复杂度为O(n' + m')，其中n'和m'分别为候选集节点数、候选集边数。n'有明显上界：$V_{\ge k}= \{v | deg_G(v) \ge k \}$。

论文中定理4和引理5略去（对m'的估计）。

在增加C的过程中，从队列选取节点的过程是随机的。论文从该点出发，提出启发式算法，使用优先队列对算法进行改进：

**优良度增量最大 Largest increment of goodness, lg**  
定义 
$$f(v) = \delta(G[C\cup \{v\}]) - \delta (G[C])$$
，f(V)不会超过1。

**连接数最大 Largest number of incidence, li**
选择与已选定点连接数最多的点。优先级函数为：
$$f(v) = deg_{G[C \cup \{v\}]}(v)$$

每次将节点添加到队列中，要执行d次更新操作（d为节点度数），每次更新需要log n'的复杂度，由此，该算法复杂度为O(n'+m' log n')。  
对于li，若使用表对相同度数节点存储于同一行，则更新操作每次需要O(1)复杂度，由此算法复杂度可以为O(m'+n')

还有一种优化方式，若用邻接列表存储每个节点的邻居，可以依照度数对邻居进行从大到小排序，再加入节点时，一旦出现邻居度数小于k，就可以立马终止对邻居的遍历操作。

### CSM的局部搜搜