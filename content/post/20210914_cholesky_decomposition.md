---
{
  "date": "2021-09-14",
  "slug": "20210914_cholesky_decomposition",
  "tags": [
    "data science"
  ],
  "title": "Data Science Concept: Cholesky Decomposition",
  "toc": "true"
}
---
<!--more-->
## Definition

A decomposition of a Hermitian, positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose:

 $ A = L  L^T$
 

*What does this mean?* Nothing more than what is stated above: a matrix factorization, albeit a very efficient one.

## Applications

### **1) Solving a linear system of equations.**

If we want to solve for $x$ in $Ax = b$, first define:

&nbsp;&nbsp;&nbsp;&nbsp; $LL^Tx=b$

&nbsp;&nbsp;&nbsp;&nbsp; $Lc=b$

From here we can solve efficiently for $c$ via *forward substitution*, and finally solve for $x$ via *backward substitution*:

&nbsp;&nbsp;&nbsp;&nbsp; $L^Tx=c$

There are other ways of doing it, but this method is highly efficient.

### **2) Solving a linear regression.**

Similar to the case above, we want to solve for $\beta$ on $Y = X \beta$. 

First we derive the least-squares solution for this problem:

&nbsp;&nbsp;&nbsp;&nbsp; $min(SSR(\beta)) = min(Y-X \beta)^T(Y-X \beta)$

&nbsp;&nbsp;&nbsp;&nbsp; $\frac{\delta SSR(\beta)}{\delta \beta} = -2X^T(Y-X \beta)=0$

&nbsp;&nbsp;&nbsp;&nbsp; $XX^T \beta = X^T Y$

Applying the Cholesky decomposition on $XX^T$:

&nbsp;&nbsp;&nbsp;&nbsp; $L^T L \beta = X^T Y$

Using a similar logic as on **(1)**, we can solve for $\beta$ via:

&nbsp;&nbsp;&nbsp;&nbsp; $Lc = XY$

&nbsp;&nbsp;&nbsp;&nbsp; $L^T \beta = c$

### **2) Create correlated variables in Monte Carlo simulation.**

This is one of the more interesting use cases. Suppose we want to transform a set of normal IID variables into correlated ones. For the case of 2 variables we know that:

&nbsp;&nbsp;&nbsp;&nbsp; $e_1 = x_1$

&nbsp;&nbsp;&nbsp;&nbsp; $e_2 = \rho x_1 + x_2\sqrt{(1-\rho^2)}$

For a larger number of variables the relationship gets tricky. However we can design a correlation matrix $\Sigma$, perform Cholesky decomposition $\Sigma=RR^T$, and then generate correlated series as:

&nbsp;&nbsp;&nbsp;&nbsp; $e = Rx$

## Example: Time Series Generation

To be less abstract, lets look at a concrete example. We will use numpy to generate time series for 3 uncorrelated variables plus a target correlation matrix. Then with Cholesky's help we will derive the correlated time series.

## Comments
- On its own the Cholesky decomposition is simply a matrix manipulation technique. The procedure itself not that interesting; what is interesting is what we can do with the transformed data representation. 
