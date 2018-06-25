# truncatedDistribution

Truncated Distributions in native [TensorFlow](https://www.tensorflow.org/). Provides truncated variates of [TensorFlow](https://www.tensorflow.org/) distributions.

The class [TruncatedDistribution](TruncatedDistribution) extends any existing TensorFlow distribution, i.e. inheriting from [tf.distribution](https://www.tensorflow.org/api_docs/python/tf/distributions/Distribution)), to enable their truncated variates, with full support of broadcasting.

See [bellow](#truncateddistribution-1) for documentation.

### Example: Truncated Gamma

```python
import tensorflow as tf
import numpy as np
from truncatedDistribution import TruncatedDistribution as TD

tf.InteractiveSession()
concentration=40.
rate=4.
gamma=tf.distributions.Gamma(concentration,rate)
left=9.
right=30.
td=TD(gamma,left,right)
samples=td.sample(1000).eval()
samples_org=gamma.sample(1000).eval()
```

![gamma_ex](/imgs/gamma_ex.png)

**Difference in statistics:**

```python
tG.mean().eval()
tG.variance().eval()
```
> 10.708002  
1.4435476

```python
gamma.mean().eval()
gamma.variance().eval()
```
> 10  

2.5


### Example: Truncated Beta


```python
a=2.
b=5.
beta=tf.distributions.Beta(a,b)
left=0.35
right=1.
tB=TD(beta,left,right)
X=np.linspace(0,1,100,dtype=np.float32)
Y1=tB.cdf(X).eval()
Y2=beta.cdf(X).eval()
```

![beta_ex](/imgs/beta_ex.png)

**Difference in statistics:**

```python
tB.mean().eval()
tB.variance().eval()
```
> 0.47338647  
0.010388413

```python
gamma.mean().eval()
gamma.variance().eval()
```
> 0.2857143  
0.025510205



## TruncatedDistribution

Truncated Distributions in native [TensorFlow](https://www.tensorflow.org/). Provides truncated variates of [TensorFlow](https://www.tensorflow.org/) distributions.

  **Attributes**:
  * dist: an instance of tf.distributions
      * (ex. Gamma, Dirichlet, etc.)
  * left: left truncation point
      * n-dimensional Tensor
      * should be compatible with dist.batch_shape, as usual
  * right: left truncation point
      * n-dimensional Tensor
      * should be compatible with dist.batch_shape, as usual
  * lft: cdf at left truncation point
      * n-dimensional Tensor
  * rght: cdf at right truncation point
      * n-dimensional Tensor
  * dist: tensorFlow distribution
  * batch_shape: batch shape of the distribution

## Methods:

### \_\_init\_\_(dist,left,right, n_points=1000)

Construct the truncated variate of a TensorFlow distribution.

  **Args**:
  * dist: an instance of tf.distributions
      * (ex. Gamma, Dirichlet, etc.)
  * left: left truncation point
      * n-dimensional Tensor
      * should be compatible with dist.batch_shape, as usual
  * right: left truncation point
      * n-dimensional Tensor
      * should be compatible with dist.batch_shape, as usual
  * n_points: number of points used for estimation of inv_cdf
      * defaults to 1000

### sample(sample_shape=())

Generates samples from the distribution.

  **Args**:
  * sample_shape: shape of the batch
    * defaults to (), ie. shape of the dist

  **Returns**:
  * a batch of samples
    * n dimensional Tensor

### cdf(X)

Cumulative distribution function.

  **Args**:
  * X: n dimensional Tenor
  
  **Returns**:
  * cdf: cdf at X

### log_cdf(X):

Logarithm of cumulative distribution function.
    
  **Args**:
  * X: n dimensional Tenor
  
  **Returns**:
  * cdf: cdf at X
    * n dimensional Tensor

### prob(X)

Probability density function
    
  **Args**:
  * X: n dimensional Tenor
  
  **Returns**:
  * pdf: pdf at X
    * n dimensional Tensor

### log_prob(X)

Logarithm of the probability density function
    
  **Args**:
  * X: n dimensional Tenor
  
  **Returns**:
  * log\_pdf: log_pdf at X
    * n dimensional Tensor

### mean(n_samples=1000)

Empirical mean of the distribution.
    
  **Args**:
  * n_samples: number of samples used
  
  **Returns**:
  * empirical mean
    * n dimensional Tensor

### variance(n_samples=1000)

Empirical variance of the distribution.
    
  **Args**:
  * n_samples: number of samples used
    * defaults to 1000
  * ddof: degrees of freedom
    * defaults to 1
  
  **Returns**:
  * empirical variance
    * n dimensional Tensor

### std(n_samples=1000)

Empirical standard deviation of the distribution.
    
  **Args**:
  * *args: arguments to be passed to self.var
  * **kwargs: names arguments to be passed to self.variance
  
  **Returns**:
  * empirical standard deviation
    * n dimensional Tensor