---
title: "Generalization Error"
date: 2024-12-06
---

# Generalization Error
## Generalization of Finite hypothesis space 
### Generalization for fixed f
**Lemma**: High probability bounds for fixed f  

$$  
Pr(Er_{in}-Er_{out} >= t) <= e^{-2nt^2}  
$$

* proof 1: 证明 $E(Er_{in})=Er_{out}$
* proof 2: apply Hoeffding‘s inequality
Xi is independent random with mean ui and bounded on [ai,bi]

$$  
Pr[\sum_{i=1}^n(Xi-ui) >= t] <= e^{-\frac{2t^2}{\sum_{i=1}^n(bi-ai)^2}}
$$

**Proposition** generalization bound for fixed f  
the following bound holds with probability $1-\delta$:  

$$  
Er_{out} <= Er_{in} + \sqrt{\frac{log(1/\delta)}{2n}}  
$$  

* proof apply the above lemma, let $\delta = e^{-2nt^2}$ , it yeilds the generalization bound.

### Generalization for finite hypothesis space  
Uniform Bound for all posibile f with probability with $1-\delta$   :  

$$  
Er_{out} <= Er_{in} + \sqrt{\frac{log(|H|/\delta)}{2n}}    \forall f \in h
$$  

**proof**  
1. 等价于证明 存在一个f满足 $Er_{in} <= Er_{out} - \sqrt{\frac{log(\|H\|/\delta)}{2n}}$ 的概率 < $\delta$, 
2. apply union bound（非常宽松，单个的概率*\|H\|） and generalization bound for a fixed f，可得概率为 $\|H\|e^{-2nt^2}$
3. setting $\|H\|e^{-2nt^2} = \delta$ yields the desired result
   
**comment**
*  from a f to finite space H, the error bound $\sqrt{\frac{log(1/\delta)}{2n}}$  changes to $\sqrt{\frac{log(\|H\|/\delta)}{2n}}$
*  the genralization error increases when \|H\| grows, but only logrithmically
*  H is usually infinite and union bound can be very coarse

## Generalization of Infinite hypothesis space  
> replace \|H\| with VC dimension
> 
> only discuss binary classification

### Growth function  
1. **Dichotomy** 对一组有限样本n的分类结果
2. **Dichotomies of H** H中所有f产生的互斥分类结果的集合（可以想到如果f有n个，分类结果可能会少于n）
3. **growth function** G（n） 考虑所有可能的n个样本的组合，选出具有最多Dichotomies的数量即为G(n)
**comment**
* Growth function measure the richness of H，based on n points rather than cardinality of H
* it's impracical to calculate G exactly for every n
* we know $2^n$ is a upper bound for G(n), but it's trivial due to log it will be a constant which means larger n cannot bring better generalization
* can we find another tighter one？--Yes！vc dimension

### break point and VC dimension 
1. **shatter**
2. break point: 不能被shatter的sample size
3. VC dimension：最大的能被shatter的sample size

**Theorm**
for d-dimensional binary linear classier, we have : dvc = d + 1   
* it's exactly the effective number of parameters, and hence the complexity of H

$$ 
G(n)<= n^(d_{VC})+1  
$$

### VC generalization bound  

$$  
O(\sqrt{\frac{dvc}{n}})  
$$  

**comment**
1. it's a litle bit loose, but establish the feasibility of learning for infinite H
2. once n is enough large, learning is likely feasible
3. give us some rules to garantee the bound: e.g. about the number of samples:n >= 10dvc
for linear regression

$$  
O(\sqrt{\frac{dp}{n}})  
$$ 

- [] todo

# Overfitting







