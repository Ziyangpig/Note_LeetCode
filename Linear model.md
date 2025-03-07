---
title: "Linear model"
date: 2024-12-06
---

# Logistic Regression
> use sigmod or softmax to approximate posterior probability
## Binary classification
**logistic function/sigmod**

$$  
h(t)=\frac{e^t}{1+e^t}=\frac{1}{1+e^{-t}}  
$$   

![sigmod 函数图像](https://img2018.cnblogs.com/blog/790418/201811/790418-20181107181130984-1052306153.png)  

## LR model
1. sigmod: $\theta x \Rightarrow probability$
   
$P(y=1\|x) = \frac{1}{1+e^{-\theta x}}$    $P(y=-1\|x) = 1 - P(y=1\|x) = \frac{1}{1+e^{\theta x}}$

$\Rightarrow$ 恰好可以合并成 $P(y\|x) = \frac{1}{1+e^{-y\theta x}}$  
2. MLE
   * likelihood = $\prod_{i=1}^n P(yi\|xi)$
   * 思考：为什么用MLE,概率乘积，而不直接max $\sum \theta x_i$ : 用乘积不会出现牺牲某个样本的概率特别小，要保证总体的预测概率都相对大，乘积才能大
3. log

$$  
max -\sum_{i=1}^nlog(1+e^(-y_i\theta x-i))  
$$  

$\Rightarrow$ 

$$  
min \frac{1}{n}\sum_{i=1}^nlog(1+e^(-y_i\theta x-i))  
$$  

## Multi-class LR
**softmax**  
1. softmax get probability
   
$$  
P(y_i=k\|x_i) = \frac{e^{\theta_k x_i}}{\sum_{j=1}^Ke^{\theta_j x_i}}  
$$

2. MLE
**注意I作为P的次数**

$$  
L(\theta) = \prod_{i=1}^n \prod_{k=1}^K P(y_i=k\|x_i)^{I(y_i = k)}
$$ 

3. log
   
$$  
min -\frac{1}{n}\sum_{i=1}^n\sum_{k=1}^K I(y_i = k) P(y_i=k\|x_i)  
$$ 

