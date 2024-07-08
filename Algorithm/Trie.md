# 字典树

### 1. 图例
![字典树示例](./pic/trie1.png "Trie example")

图中为字典树的一个例子，字典树实际上为一个26叉树。对于26叉树的每一个节点，我们可以维护一个映射来记录跳转关系。如f(1,a)=2, f(1,b)=3, f(1,c)=4。

### 2. 字典树的实现
``` cpp
struct trie {
    int nex[100000][26]; // 26叉树节点间的映射关系
    int cnt;    // 记录nex数组下一个可作为树节点的位置
    bool exist[100000];  // 该结点结尾的字符串是否存在

    void insert(char *s, int l) {  // 插入字符串
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!nex[p][c]) 
                nex[p][c] = ++cnt;  // 如果没有，就添加结点
            p = nex[p][c];
        }
        exist[p] = 1;
    }

    bool find(char *s, int l) {  // 查找字符串
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!nex[p][c]) return 0;
            p = nex[p][c];
        }
        return exist[p];
    }
};
```


