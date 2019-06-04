---
title: 多模态融合在用户画像的应用
# image: /images/black_hawk.png
categories:
  - Multimodal Fusion
tags:
  - User Profiling
last_modified_at: 2019-06-03T13:57:52-17:00
---

### 背景 
   以往我们在优化画像标签的时候，一版只会用到一些结构化的特征，比如：用户的浏览/下单的行为。但是，很少用到非结构化特征，比如：图片，文本，语音等。而这些非结构化的特则却能给画像标签的构建带来很大的帮助。例如，用户的注册头像可以帮助我们判断用户的性别、用户的社交人群可以帮助我们判断用户的职业、收入水平等。    
论文《User Profiling through Deep Multimodal Fusion》中就介绍了通过深度多模态融合（UDMF）进行用户分析的方法

### UDMF介绍

- 数据源   
    主要包含：
    - 文本数据
    - 图像数据：CNN提取
    - 社交关系数据（用户喜欢的页面）：Node2vec 提取      
- early/late approach   
    - early approach:在特征级别（feature level）就融合各种数据源   
    优点：可以考虑到不同数据源之间的关系  
    缺点：有时不同数据源的特征之间缺乏直接的相关性，很难进行融合，比如：用户发布的图片 和 博客。 
    - late approach: 在决策级别（decision level）进行融合,最简单的方式就是线性加权   
- stacking  
    学习的目标之间存在相互关联。 
    使用1个数据源，且用相同的网络结构以两个目标进行建模，每个网络的输入包括输入的数据源和另一个目标变量的预测输出。如下图所示：    
    ![img]({{'images/picture/UPMF/upmf-01.png' | relative_url}})    
    在原本的公式：  
    $$ U_{i}^{h}(D) = f(\sum_{j}w_{ij}^{hl}\cdot U_{j}^{l}(D)) $$
    其中h表示当前层，l表示下一层，i表示h层的节点，j表示l层的节点，w表示连接h层和l层的权重，f表示激活函数。    
    当h为第0层的时候，公式如下：     
    $$ U_{i}^{0}(D) = f(\sum_{j}w_{ij}\cdot D_{j}) $$   
    在加入staking思想之后，公式变为如下形式：  
    $$ U_{i}^{0q}(D) = f(\sum_{j}w_{ij}\cdot D_{j} + \sum_{z}w_{iz}\cdot \alpha_{z}\cdot t_{z}^{q-1}) $$   
    $$ \alpha_{z} $$ 的取值范围为（0，1），当z的取值和目标一致时，$$ \alpha_{z}=0 $$,否则$$ \alpha_{z}=1 $$, $$ t_{z}^{q-1} $$是另一个网络在第q-1个epoch的预测结果值。q=0时，$$ t_{z}^{q-1} $$。  
    $$ U_{0}^{0}(D) = f(\sum_{j}w_{ij}\cdot D_{j}) $$   
    每一轮epoch目标变量的预测值在以另一个目标建模的网络前一步的预测值的基础上进行更新。ps：（不一定每一个epoch更新一次，可以10个epoch更新1次，也不一定只有连个网络，可以有更多的网络） 
- Power-set Combination
    $$  DS={D_{1},D_{2},...,D_{k}} $$表示k个数据源，例如：k可以表示5种数据源：文本，视频，关系，时间，定位。假设有两个数据源 a 和 b，那么就可以产生a,b,ab,{}这4中子数据集，可以训练3个mini-dnn，udmf的结构入下图所示。     
    ![img]({{'images/picture/UPMF/upmf-02.png' | relative_url}})    
    有2个target，2个数据源（$$ 2^{k-1} $$个子数据集=3），所以一共有2*3=6个mini-dnn。每个mini-dnn的输出都会作为另一个建模目标的姐妹min-dnn的下一个epoch的输入。这样做的好处是，每个网络可以单独训练，并行训练多目标大大减少了训练时间

### 实验
- 模型：使用的都是普通的3层dnn
- 数据源：
    - 文本：LIWC-based DNN
        * standard counts ：词语出现的次数
        * standard counts ：文本中表示愤怒的词语出现的次数，hate，annoyed
        * relativity : 将来时的动词数目
        *  personal concerns ：和职业相关的词的数目，job,majors
    - 视频：Oxford Face API：提取面部关键点
    - 社交：Node2Vec-Skip-gram   
- UDMF结构    
    ![img]({{'images/picture/UPMF/upmf-03.png' | relative_url}})    
    其中，T,I,R分别代表文本，图片，和关系。作者使用不同的数据源，不同的方法，不同的结果融合方法进行了实验，实验结果如下：   
    ![img]({{'images/picture/UPMF/upmf-04.png' | relative_url}})  
    
### 总结
   这篇论文使用了多模态的方法（多种数据源）来实现用户属性的预测，从实验结果上来看效果还不错，也算是做画像相关工作的一种新的方法。可以尝试融合文本数据，或者来自不同平台的数据；但是如果要融合视频，或者图片数据感觉还是过重了一些。
    
