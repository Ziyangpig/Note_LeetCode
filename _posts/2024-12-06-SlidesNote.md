
---
title: "ML-Notes"
date: 2024-12-06
---

# train versus test
## Finite hypothesis space generalization
### Generalization for fixed f
**Lemma**: High probability bounds for fixed f  

$$  
Pr(Er_{in}-Er_{out} >= t) <= e^{-2nt^2}  
$$

* proof 1: 证明 $E(Er_{in})=Er_{out}$
* proof 2: apply Hoeffding‘s inequality

**Proposition** generalization bound for fixed f  
the following bound holds with probability $1-\delta$:  

$$  
Er_{out} <= Er_{in} + \sqrt{\frac{log(2/\delta)}{2n}}  
$$  

* proof apply the above lemma, let $\delta = e^{-2nt^2}$ , it yeilds the generalization bound.

### Generalization for finite hypothesis space: Uniform Bound for all posibile f  

