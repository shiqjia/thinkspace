---
title: Deep Cross Net
# image: /images/black_hawk.png
categories:
  - ctr预估
tags:
  - dcn
last_modified_at: 2019-03-26T13:57:52-17:00
---

# Deep & Cross Network for Ad Click Predictions
google在2017年提出的Deep&Cross Network模型简介
   
## 背景
特征工程一直是许多预测模型成功的关键，但是现有模型还是存在一些不足：
- DNN在特征非常稀疏的情况下，几乎没有什么特征交叉，这时只能达到线性模型的效果
- wide&deep的wide侧还是需要进行人工的特征构造
- deepfm的fm侧只能构造特征之间的2阶关系  
因此,DCN被提出了，它能对sparse和dense的输入自动学习特征交叉，可以有效地捕获有限阶（bounded degrees）上的有效特征交叉，无需人工特征工程或暴力搜索（exhaustive searching），并且计算代价较低。  

## 网络结构
![img]({{"images/picture/DCN/DCN_model.png" | relative_url }})  

### 嵌入层
- 先对稀疏特征做embedding，并与稠密特征进行concat

### 交叉网络层
- 每一层的输出计算:  
$$ x_{l+1} = x_{0}x_{l}^{T}w_{l}+x_{l} = f(x_{l},w_{l},b_{l})+x_{l} $$
- 函数f部分刚好拟合的是$$ x_{l+1} $$和$$ x_{l} $$的残差
- 特征高阶交叉：特征的阶随着layer的深度而增长，对于第l层，它的最高阶是l+1,crossnet的结构使得它可以构造不同阶的交叉特征
- 复杂度分析：参数的维度为，$$ d×L_{C}×2 $$,其中d表示$$ x_0 $$的维度，$$ L_{C} $$表示交叉网络层的深度。一个cross network的时间和空间复杂度对于输入维度是线性关系。因而，比起它的deep部分，一个cross network引入的复杂度微不足道，DCN的整体复杂度与传统的DNN在同一水平线上。
- 实现技巧：第一眼看上面的计算公式，会认为计算$$ x_{0}x_{l}^{T} $$需要的内存大小为：batch_size×d×d×4，但是如果先计算$$ x_{l}^{T}w_{l} $$会得到一个标量，代码参考如下：
    
      def cross_layer2(x0, x, name):
        with tf.variable_scope(name):
        input_dim = x0.get_shape().as_list()[1]
        w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
        return x0 * xb + b + x
      
### 深度网络
这里就是使用了dnn,用于获取高阶非线性交叉


### Combination layer
将cross network以及deep network的输出结果进行concat，然后将结果送入logits layer（标准逻辑回归）

## 总结
- 一种新型的交叉网络结构，可以用来提取交叉组合特征，并不需要人为进行特征工程
- 随着网络层数的增加，可以构造多项式阶的交叉特征
- 时间和空间复杂度对于输入维度是线性关系，相对于DNN，没有增加参数量级
- 与DC,DNN,FM,LR相比有更好的结果

## 参考
博客：https://xudongyang.coding.me/dcn/