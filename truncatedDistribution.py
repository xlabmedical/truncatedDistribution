"""Žiga Sajovic
"""
import tensorflow as tf


class TruncatedDistribution:

  """Truncated Distributions in native TensorFlow. Provides truncated variates of TensorFlow distributions.

    Attributes:
      * dist: an instance of tf.distributions
          * (ex. Gamma, Dirichlet, etc.)
      * left: left truncation point
        * a scalar or an n-dimensional Tensor
        * should be compatible with dist.batch_shape, as usual
      * right: right truncation point
        * a scalar or an n-dimensional Tensor
        * should be compatible with dist.batch_shape, as usual
      * lft: cdf at left truncation point
          * n-dimensional Tensor
      * rght: cdf at right truncation point
          * n-dimensional Tensor
      * dist: tensorFlow distribution
      * batch_shape: batch shape of the distribution
    """

  def __init__(self, dist, left, right, n_points=1000):
    """Construct the truncated variate of a TensorFlow distribution

    Args:
      * dist: an instance of tf.distributions
        * (ex. Gamma, Dirichlet, etc.)
      * left: left truncation point
        * a scalar or an n-dimensional Tensor
        * should be compatible with dist.batch_shape, as usual
      * right: right truncation point
        * a scalar or an n-dimensional Tensor
        * should be compatible with dist.batch_shape, as usual
      * n_points: number of points used for estimation of inv_cdf
        * defaults to 1000
    """
    left = tf.convert_to_tensor(left)
    right = tf.convert_to_tensor(right)
    self.lft = dist.cdf(left)
    self.rght = dist.cdf(right)
    shape = tf.shape(self.lft) \
        if self.lft.shape.ndims >= self.rght.shape.ndims \
        else tf.shape(self.rght)
    self.left = tf.ones(shape)*left
    self.right = tf.ones(shape)*right
    shape = tf.shape(self.left) \
        if self.left.shape.ndims >= self.right.shape.ndims \
        else tf.shape(self.right)
    self.yaxis = tf.reshape(
        tf.map_fn(
            lambda pt: tf.linspace(pt[0], pt[1], n_points),
            tf.stack(
                [tf.reshape(self.left, [-1]), tf.reshape(self.right, [-1])],
                axis=1)),
        tf.concat([[n_points], shape], axis=0))
    self.xaxis = dist.cdf(self.yaxis)
    self.dist = dist
    self.batch_shape = dist.batch_shape

  def sample(self, sample_shape=()):
    """Generates samples from the distribution.

    Args:
      * sample_shape: shape of the batch
        * defaults to (), ie. shape of the dist
    Returns:
      * a batch of samples
        * n dimensional Tensor
    """
    if sample_shape is ():
      sample_shape = (1)
    sample_shape = tf.convert_to_tensor(sample_shape)
    if sample_shape.shape.ndims == 0:
      sample_shape = tf.expand_dims(sample_shape, 0)
    sample_shape_original = sample_shape
    sample_shape = tf.reduce_prod(tf.reshape(sample_shape, [-1]))
  #
    to_sample = tf.concat(([sample_shape], tf.shape(self.xaxis)[1:]), axis=0)
    query = self.lft+tf.contrib.distributions.Uniform().sample(to_sample) * \
        (self.rght-self.lft)
  #
    l_pad = tf.expand_dims(tf.minimum(query - 1, self.xaxis[0]), axis=0)
    r_pad = tf.expand_dims(tf.maximum(query + 1, self.xaxis[-1]), axis=0)
  #
    to_tile_x = tf.concat(
        [[1], [sample_shape],
         tf.ones(
            tf.size(tf.shape(self.xaxis)[1:]),
            dtype=tf.int32)],
        axis=0)
    mid_pad = tf.tile(tf.expand_dims(self.xaxis, 1), to_tile_x)
    xs_pad = tf.concat([l_pad, mid_pad, r_pad], axis=0)
  #
    ys_pad = tf.concat(
        [self.yaxis[:1], self.yaxis, self.yaxis[-1:]],
        axis=0)
    ys_pad = tf.tile(tf.expand_dims(ys_pad, 1), to_tile_x)
    ys_pad = tf.tile(ys_pad, tf.shape(xs_pad)//tf.shape(ys_pad))
  #
    perm = tf.concat(
        [tf.range(1, tf.size(tf.shape(xs_pad)), dtype=tf.int32), [0]],
        axis=0)
    ys_pad = tf.transpose(ys_pad, perm)
    xs_pad = tf.transpose(xs_pad, perm)
    query = tf.expand_dims(query, -1)
  #
    cmp = tf.transpose(tf.cast(query >= xs_pad, dtype=tf.int32))
    diff = tf.transpose(cmp[1:] - cmp[:-1])
    idx = tf.argmin(diff, axis=-1, output_type=tf.int32)
  #
    shape = tf.shape(xs_pad)
    idx_flat = tf.range(tf.reduce_prod(
        shape[0:-1]))*shape[-1]+tf.reshape(idx, [-1])
    idx1_flat = idx_flat+1
  #
    xpad_flat = tf.reshape(xs_pad, [-1])
    ypad_flat = tf.reshape(ys_pad, [-1])
  #
    xps = tf.gather(xpad_flat, idx_flat)
    xps1 = tf.gather(xpad_flat, idx1_flat)
    yps = tf.gather(ypad_flat, idx_flat)
    yps1 = tf.gather(ypad_flat, idx1_flat)
  #
    alpha = (tf.reshape(query, shape[:-1]) - tf.reshape(xps,
                                                        shape[:-1])) / tf.reshape(xps1 - xps, shape[:-1])
    res = alpha * tf.reshape(yps1, shape[:-1]) + \
        (1 - alpha) * tf.reshape(yps, shape[:-1])
    return tf.squeeze(
        tf.reshape(
            res, tf.concat(
                [sample_shape_original, tf.shape(res)[1:]], axis=0)))

  def cdf(self, X):
    """Cumulative distribution function.

    Args:
      * X: n dimensional Tensor
    Returns:
      * cdf: cdf at X
    """
    X = tf.maximum(tf.minimum(X, self.right), self.left)
    return (self.dist.cdf(X)-self.lft)/(self.rght-self.lft)

  def log_cdf(self, X):
    """Logarithm of cumulative distribution function.

    Args:
      * X: n dimensional Tensor
    Returns:
      * cdf: cdf at X
        * n dimensional Tensor
    """
    X = tf.maximum(tf.minimum(X, self.right), self.left)
    return tf.log((self.dist.cdf(X)-self.lft)/(self.rght-self.lft))

  def survival_function(self, X):
    """Survival function.

    Args:
      * X: n dimensional Tensor
    Returns:
      * survival_function: 1 - cdf at X
    """
    return 1. - self.cdf(X)

  def log_survival_function(self, X):
    """Logarithm of the Survival function.

    Args:
      * X: n dimensional Tensor
    Returns:
      * log survival_function: log(1 - cdf) at X
    """
    return tf.log(1. - self.cdf(X))

  def prob(self, X):
    """Probability density function

    Args:
      * X: n dimensional Tensor
    Returns:
      * pdf: pdf at X
        * n dimensional Tensor
    """
    mask = (X >= self.left)*(X <= self.right)
    return self.dist.prob(X)*mask/(self.rght-self.lft)

  def log_prob(self, X):
    """Logarithm of the probability density function

    Args:
      * X: n dimensional Tensor
    Returns:
      * log_pdf: log_pdf at X
        * n dimensional Tensor
    """
    mask = (X >= self.left)*(X <= self.right)
    return tf.log(self.dist.prob(X)/(self.rght-self.lft))*mask

  def mean(self, n_samples=1000):
    """Empirical mean of the distribution.

    Args:
      * n_samples: number of samples used   
    Returns:
      * empirical mean
        * n dimensional Tensor
    """
    return tf.reduce_mean(self.sample(n_samples), axis=0)

  def variance(self, ddof=1, n_samples=1000):
    """Empirical variance of the distribution.

    Args:
      * n_samples: number of samples used
        * defaults to 1000
      * ddof: degrees of freedom
        * defaults to 1   
    Returns:
      * empirical variance
        * n dimensional Tensor
    """
    samples = self.sample(n_samples)
    return tf.reduce_sum((samples-tf.reduce_mean(samples))**2, axis=0)/(n_samples-ddof)

  def stddev(self, *args, **kwargs):
    """Empirical standard deviation of the distribution.

    Args:
      * *args: arguments to be passed to self.var
      * **kwargs: names arguments to be passed to self.variance 
    Returns:
      * empirical standard deviation
        * n dimensional Tensor
    """
    return tf.sqrt(self.variance(*args, **kwargs))
