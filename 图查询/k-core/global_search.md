# 论文 The Community-search Problem and How to Plan a Successful Cocktail Party 笔记
2010

### 问题定义

#### Problem 1 (Generic objective function:) 
给无向完全图$G=(V,E)$，查询点集$Q \subseteq V$，优良度评价函数$f$(goodness function)，找到一个诱导子图$H=(V_H, E_H)$，使得：  
(i) $V_H$ 属于$Q(Q\subseteq V_H)$；  
(ii) $H$ 连通；  
(iii) $f(H)$在所有可行的$H$选择中最大。

在本文中，将关注$f_m$，$f_m(H)$表示$H$中所有节点度数的最小值，它有助于防止并入离$Q$很远的社区。

另一种防止并入离查询节点很远的社区的方法，为设置距离约束。  
设$d_G(v,q)$ 为$G$中$d$与$q$最短路径的距离。则  
$D_Q(G, v) = \sum_{q\in Q} d_G(v,q)^2$，用节点v离Q中所有查询节点的距离的平方和表示v到Q的距离；  
$D_Q(G) = max_{v\in V(G)}\{D_Q(G, v)\}$，用最大的v到Q距离值作为图的一个距离属性。  

由此，可以定义第二类问题：
#### Problem 2
给无向完全图$G=(V,E)$，查询点集$Q \subseteq V$，距离限制参数$d$，找到一个诱导子图$H=(V_H, E_H)$，使得：  
(i) $V_H$ 属于$Q(Q\subseteq V_H)$；  
(ii) $H$ 连通；  
(iii) $D_Q(H) \le d$；  
(iv) $f_m(H)$在所有可行的$H$选择中最大。

### 无大小约束的社区