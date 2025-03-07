---
title: "Optimization Algorithm and Convergence"
date: 2024-12-07
---


# iterative algorithm
## elements
* initial point(fine for convex problem)
* search direction
* step size/ learning rate
* stopping criterion

## A useful Algorithm Design Framework
suppose the task is min L, then we design a framework as:

$$  
\theta_{k+1} = min\{q_{k}+\frac{1}{2u_k}\|\theta-\theta_k\|_2^2\}
$$

**comment**   
cannot directly minimize the first approximation term, which is accurate only around $\theta_k$, thus we need the proximal term, which can pinish a large step

* when $q_k$ is **linear approximation** of L $Rightarrow$ **gradient descent**
* when $q_k$ is **second-order approximation** of L $Rightarrow$ **Newton's method**
* when $q_k$ is L $Rightarrow$ proximal point method(solving nonsmooth problem)
* when $q_k$ is single component linear approximation of L $Rightarrow$ stochastic gradient method

**almost universal algorithmic design strategy**
solving the original problem by solving a sequence of simpler subproblems

# Choice of search direction
## Gradient-based Optimization Algorithm

> most learning problems do not have closed form solution, but they are convex problems, which provide great property when using gradient-based methods
>
> ex. Logistic regression, SVM, ...,
> 
> namely, the point where $\{nbala}=0$ is the global optimal point

### GD
more info will be shown in optimization part! 
- [] to be down

### Gradient Descent with momentum

                                                                                                                                               
## convergence
if L is convex and have Lipschitz smooth parameter L, then gradient descent with constant learning rate 1/L satisfies:

$$
L(\theta_k)-L(\hat{\theta}) \leq \frac{L\|(\theta_0-\hat{\theta})\|_2^2}{2k}
$$

* 次线性收敛速度： $\cal O(1/k)$
* does not mean $\theta_k$ converge to $\theta$ at a certain rate




**supplementary**  
两种收敛速度的判别标准：Q-收敛速度 和 R-收敛速度
* R-收敛速度  $\|x_{k+1} - \hat{x}\|$
  * 几何收敛速率/超线性收敛：  $\|x_{k+1} - \hat{x}|\ \leq \rho\|x_k - \hat{x}\|$
  * 线性收敛：
  * 次线性收敛
* R-收敛速度  $\frac{\|x_{k+1} - \hat{x}\|}{\|x_{k} - \hat{x}\|}$

  [收敛速度](https://blog.csdn.net/qq_34758157/article/details/132081279)
  
*   
# Choice of Learing rate and Stopping criterion
learing rate。

* constant
* line search
* decaying learning rate
stopping criterion
* fix the number of iterations as K
* 梯度足够小
* $\theta_{k+1} - \theta_k$ 足够小
* in ML, stop when val error is gonna increase
  

