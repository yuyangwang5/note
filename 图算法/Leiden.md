## From Louvain to Leiden: guaranteeing well-connected communities 2018

### 社区质量函数

#### 模块度，Modularity
该方法尝试将社区内实际边数和预期边数（假设边随机均匀生成）之间的差距最大化。设$e_c$表示社区c中的边数，$\frac{K_{c}^{2}}{2m}$表示预期的边数，其中$\K_{c}^{2}$表示社区c中的节点度数之和，m表示网络中的边数。由此，模块度的公式为：

$$\mathcal{H} = \frac{1}{2m} \sum_c(e_c - \gamma \frac{K_{c}^{2}}{2m})$$

$\gamma \ge 0$是一个分辨率参数，高分辨率会有更多社区，低分辨率则相反。

#### CPM, Constant Potts Model
CPM函数定义为：

$$\mathcal{H} = \sum_c(e_c - \gamma 
\left(
    \begin{matrix}
    n_c \\
    2
    \end{matrix}
\right)
)$$

其中$n_c$表示社区c中的节点数。分辨率参数$\gamma$在该公式中有直观的解释：社区内部的密度应不少于$\gamma$，而社区间应低于$\gamma$。更高的$\gamma$会带来更多社区，低则相反。

### Louvain
该算法通过两个基本阶段来优化质量函数：  
1. 节点的局部移动；
2. 网络的聚合。
   
在局部移动阶段，单个节点被移动到能使质量函数增加到最大的社区中；在聚合阶段，将会根据第一阶段的结果对网络进行分区，之后每个分区视为一个节点，由此可以看成一个新的网络。  
这两个阶段不断重复，直到任何局部移动都无法增加质量函数。  
该算法流程可用下图表示：

![example of louvain](./Leiden_pic/1%20example%20of%20louvain.png "example of louvain")

一般，最开始时每个节点看成一个社区。不过也可以用不同分区方式进行初始化。如，在尝试找更好的分区时，可以先执行算法，进行多次迭代，使用其中一个迭代所识别的分区进行初始化。

然而Louvain有个缺点：找到的分区可能内部连接性很差，甚至会识别出内部断开的社区。在实验中，断开的社区还经常出现。不断的迭代，虽然会增加函数质量，但会加剧这种社区内部断裂的问题。在Louvain算法中，一个节点被移动到另一个社区时，这个节点在原社区中，可能是重要的连接节点，如下图所示：
![disconnected community](./Leiden_pic/2%20disconnected%20community.png "disconnected community")

这个问题出在Louvain算法本身，与质量函数无关。若只考虑社区内部断开这种极端表现，无法解决根本问题，需要更进一步考虑。

### Leiden






