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

