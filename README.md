# truncatedDistribution

Truncated Distributions in native [TensorFlow](https://www.tensorflow.org/). Provides truncated variates of [TensorFlow](https://www.tensorflow.org/) distributions.

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

### Methods

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

### empirical\_mean(n_samples=1000)

Empirical mean of the distribution.
    
  **Args**:
  * n_samples: number of samples used
  
  **Returns**:
  * empirical mean
		* n dimensional Tensor

### empirical\_var(n_samples=1000)

Empirical variance of the distribution.
    
  **Args**:
  * n_samples: number of samples used
		* defaults to 1000
  * ddof: degrees of freedom
		* defaults to 1
  
  **Returns**:
  * empirical variance
		* n dimensional Tensor

### empirical\_std(n_samples=1000)

Empirical standard deviation of the distribution.
    
  **Args**:
  * *args: arguments to be passed to self.empirical_var
  * **kwargs: names arguments to be passed to self.empirical_var
  
  **Returns**:
  * empirical standard deviation
		* n dimensional Tensor