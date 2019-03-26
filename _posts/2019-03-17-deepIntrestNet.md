---
title: Deep Interest Net
# image: /images/black_hawk.png
categories:
  - ctr预估
tags:
  - din
last_modified_at: 2019-03-17T13:57:52-17:00
---

# Deep Interest Network for Click-Through Rate Prediction  
阿里2018年广告点击率预估的论文，记录论文主要思想以及用到的一些技巧。

## 背景

### 基础模型
#### 特征表达
$$t_i$$ 表示低i个特征组，$$K_i$$表示特征组i的维度(特征组i包含K个唯一的可能取值)，$$t_i[j]$$ 表示特征组i的第j个元素,$$\sum_{j=1}^{K_i}t_i[j] = k$$当k=1时为one-hot向量，当k>1时为muti-hot向量。一条样本数据可以被表示成：$$x = [t_{1}^{T},t_{2}^{T},....,t_{M}^{T}]$$，其中M表示特征组的个数。图示如下：  
![image]({{'images/picture/DIN/DIN_feature.png'| relative_url}})
#### 模型结构
![image]({{'images/picture/DIN/DIN_baseModel.png'| relative_url}})  
- embedding layer
    - 将高维向量映射到低维。如果特征是one-hot的，那么得到一个单独的embedding向量为ei；如果特征是muti-hot的，得到embeding向量组成的list,$$[e_{i_{1}},e_{i_{2}},...,e_{i_{k})$$
- Pooling layer and Concat layer
    - 由于每个用户muti-hot的特征不同，embeding向量长度也不同
    - 所以要进行pooling操作，一般是max/avg pooling
    - 最后再将各个特征组的embedding拼接起来
- MLP
    - 全连接网络，用于学习特征之间的关系
- Loss
    - 交叉熵：![image]({{'/images/picture/DIN/DIN_lossFuction.png' | relative_url}}) 
    

#### 存在问题  
现在CTR常用预估方式为Embedding&MLP模式，但是这个模式存在一些问题：  

1.用户兴趣是多样化的,采用embedding压缩到固定长度会会限制模型的表达能力，例如：一个年轻母亲可能同时对手提包，连衣裙，儿童玩具感兴趣
- 解决办法：  
    1）扩展embedding字段的长度
    - 问题：问题：训练参数增加，过拟合风险增大；计算负担增大，存储参数增多，不适应线上使用。 
      
    2）不用将用户多样化的兴趣映射到同一个特征向量，因为只有用户的部分兴趣会影响他当前的行为  
    - 例如：一个游泳运动员是否点击推荐的眼镜，应该与她之前购买的泳衣有关而不是她上周加入购物车的鞋子相关
    - 根据候选集ad和用户历史行为的相关性，自适应的生成用户兴趣向量  

2.有大量稀疏特征的网络模型很难训  
    原因：  
 - 每个mini-batch出现过的稀疏特征很少，SGD只能优化部分参数
 - L2正则很难实施（参数太多,计算量大）  
    解决办法：
 - mini-bach中的非0特征的参数才用于计算L2正则
 - 根据输入数据分布，动态调整正则参数
 
## DIN  
![image]({{'images/picture/DIN/DIN_model.png'| relative_url}})  
将用户在不同商品上的行为特征embedding分别和广告特征embeding做內积，将运算结果和这两个embedding进行concat之后丢进网络模型，得到1个输出作为权重。再将这个权重分别给到对应计算的商品，再丢进网络进行运算。公式如下：  
![image]({{'images/picture/DIN/DIN_function.png'| relative_url}})  
其中，$$v_A$$表示候选广告embedding，$$e_j$$表示用户行为embedding,a(.)是一个前馈神经网络，输出为activation weight。(借鉴自attention,不再要求 $$\sum_{i}w_i =1$$，将得到的$$v_U$$ 看做是兴趣程度）
 
#### 训练技巧  
- mini-batch粒度的正则  
  ![image]({{'images/picture/DIN/DIN_regulation1.png'| relative_url}})  
  $$w_j$$表示第j个embedding向量，$$I(x_j != 0))$$表示样本x是否有特征j,$$n_j$$表示特征j在数据集中出现的次数。也可以改写成：  
  ![image]({{'images/picture/DIN/DIN_regulation1.png'| relative_url}})  
  ![image]({{'images/picture/DIN/DIN_regulation2.png' | relative_url}})  
  B表示mini-batch  
  ![image]({{'images/picture/DIN/DIN_regulation3.png'| relative_url}})  
  $$\alpha_{mj}$$表示这个batch是否至少有一个样本x，包含特征j  
  权重更新公式如下： 
  ![image]({{'images/picture/DIN/DIN_regulation4.png'| relative_url}}) 
- 激活函数Dice  
    - 激活函数relu在x<0时，梯度为0，这会导致部分参数更新缓慢，因此提出PRelu
    - 但是PRelu默认分割点为0，作者认为分割点应该由数据决定，因此提出Dice
    - dice根据输入动态调整，当E(s)=0 and Var[s]=0时，Dice=PReLU
    - ![image]({{"images/picture/DIN/DIN_Dice.png"| relative_url}})  
    
- GAUC  
 ![image]({{'images/picture/DIN/DIN_GAUC.png' | relative_url}})  
    - 假设有两个用户A和B，每个用户都有10个商品，10个商品中有5个是正样本，我们分别用TA，TB，FA，FB来表示两个用户的正样本和负样本。也就是说，20个商品中有10个是正样本。假设模型预测的结果大小排序依次为TA，FA，TB，FB。如果把两个用户的结果混起来看，AUC并不是很高，因为有5个正样本排在了后面，但是分开看的话，每个用户的正样本都排在了负样本之前，AUC应该是1。显然，分开看更容易体现模型的效果，这样消除了用户本身的差异。
    - 但是上文中所说的差异是在用户点击数即样本数相同的情况下说的。还有一种差异是用户的展示次数或者点击数，如果一个用户有1个正样本，10个负样本，另一个用户有5个正样本，50个负样本，这种差异同样需要消除。那么GAUC的计算，不仅将每个用户的AUC分开计算，同时根据用户的展示数或者点击数来对每个用户的AUC进行加权处理。进一步消除了用户偏差对模型的影响。通过实验证明，GAUC确实是一个更加合理的评价指标。
- AUC相对提升    
 ![image]({{"images/picture/DIN/DIN_relaimpr.png" | relative_url}})  
 
## 思考
- 在一些与时间相关的场景，context特征可以加入时间特征
- 思考怎么实现如下正则? 如果某个id出现的次数过少或者虽然出现很多次但是这个id加或者不加对模型没有影响, 怎么将它的embedding向量全部置0? 
    - DIN论文中加入的正则可以减少那些id出现的次数很少的特征的重要性，但是还是不能完成上述需求，考虑将L2正则换成L1正则，可以达到如上需求  
    
## 参考博客：
https://www.lfyun.com/forum/8247/
 
 
 