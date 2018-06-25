"""Žiga Sajovic
"""
import tensorflow as tf

class TruncDist:

    """Provides truncated variates of any TensorFlow distribution
    """

#
  def __init__(self,dist,left,right, n_points=1000):
    """Construct the truncated variate of a TensorFlow distribution
    
    Args:
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
    """
    self.left=left
    self.right=right
    self.lft=dist.cdf(left)
    self.rght=dist.cdf(right)
    l_shape=left.shape
    self.yaxis=tf.reshape(tf.map_fn(lambda lft: tf.linspace(lft,right,n_points), left.reshape(-1)),tf.concat([[n_points],l_shape],axis=0))
    self.xaxis=dist.cdf(self.yaxis)
    self.dist=dist
    self.batch_shape=dist.batch_shape
#
  def sample(self,sample_shape=()):
    """Samples from the distribution
    
    Args:
        * sample_shape: shape of the batch
                * defaults to (), ie. shape of the dist
    Returns:
        * a batch of samples
                * n dimensional Tensor
    """
    if sample_shape is ():
      sample_shape=(1)
    sample_shape=tf.convert_to_tensor(sample_shape)
    if sample_shape.shape.ndims == 0:
      sample_shape=tf.expand_dims(sample_shape,0)
    sample_shape_original=sample_shape
    sample_shape=tf.reduce_prod(tf.reshape(sample_shape,[-1]))
  #
    to_sample=tf.concat(([sample_shape],tf.shape(self.xaxis)[1:]),axis=0)
    query=self.lft+tf.contrib.distributions.Uniform().sample(to_sample)*(self.rght-self.lft)
  #
    l_pad=tf.expand_dims(tf.minimum(query - 1, self.xaxis[0]),axis=0)
    r_pad=tf.expand_dims(tf.maximum(query + 1, self.xaxis[-1]),axis=0)
  #
    to_tile_x=tf.concat([[1],[sample_shape],tf.ones(tf.size(tf.shape(self.xaxis)[1:]),dtype=tf.int32)],axis=0)
    mid_pad=tf.tile(tf.expand_dims(self.xaxis,1),to_tile_x)
    xs_pad = tf.concat([l_pad,mid_pad,r_pad],axis=0)
  #
    ys_pad = tf.concat([self.yaxis[:1], self.yaxis, self.yaxis[-1:]], axis=0)
    ys_pad=tf.tile(tf.expand_dims(ys_pad,1),to_tile_x)
  #
    perm=tf.concat([tf.range(1,tf.size(tf.shape(xs_pad)),dtype=tf.int32),[0]],axis=0)
    ys_pad=tf.transpose(ys_pad,perm)
    xs_pad=tf.transpose(xs_pad,perm)
    query=tf.expand_dims(query,-1) 
  #
    cmp = tf.transpose(tf.cast(query>= xs_pad, dtype=tf.int32))
    diff = tf.transpose(cmp[1:] - cmp[:-1])
    idx = tf.argmin(diff, axis=-1,output_type=tf.int32) 
  # 
    shape=tf.shape(xs_pad)
    idx_flat=tf.range(tf.reduce_prod(shape[0:-1]))*shape[-1]+tf.reshape(idx,[-1])
    idx1_flat=idx_flat+1
  #
    xpad_flat=tf.reshape(xs_pad,[-1])
    ypad_flat=tf.reshape(ys_pad,[-1])
  #
    xps=tf.gather(xpad_flat,idx_flat)
    xps1=tf.gather(xpad_flat,idx1_flat)
    yps=tf.gather(ypad_flat,idx_flat)
    yps1=tf.gather(ypad_flat,idx1_flat)
  #
    alpha = (tf.reshape(query,shape[:-1]) - tf.reshape(xps,shape[:-1])) / tf.reshape(xps1 - xps,shape[:-1])
    res = alpha * tf.reshape(yps1,shape[:-1]) + (1 - alpha) * tf.reshape(yps,shape[:-1])
    return tf.squeeze(tf.reshape(res,tf.concat([sample_shape_original,tf.shape(res)[1:]],axis=0)))
#
  def cdf(self,X):
    """Cumulative distribution function.

    Args:
        * X: n dimensional Tenor
    
    Returns:
        * cdf: cdf at X
    """
    X=tf.maximum(tf.minimum(X,self.right),self.left)
    return (self.dist.cdf(X)-self.lft)/(self.rght-self.lft)
#
  def log_cdf(self,X):
    """Logarithm of cumulative distribution function.
    
    Args:
        * X: n dimensional Tenor
    
    Returns:
        * cdf: cdf at X
    """
    X=tf.maximum(tf.minimum(X,self.right),self.left)
    return tf.log((self.dist.cdf(X)-self.lft)/(self.rght-self.lft))
#
  def prob(self,X):
    """Probability density function
    
    Args:
        * X: n dimensional Tenor
    
    Returns:
        * pdf: pdf at X
    """
    mask=(X>=self.left)*(X<=self.right)
    return self.dist.prob(X)*mask/(self.rght-self.lft)
#
  def log_prob(self,X):
    """Logarithm of the probability density function
    
    Args:
        * X: n dimensional Tenor
    
    Returns:
        * pdf: pdf at X
    """
    mask=(X>=self.left)*(X<=self.right)
    return tf.log(self.dist.prob(X)/(self.rght-self.lft))*mask
#
  def empirical_mean(self, n_samples=1000):
    """Empirical mean of the distribution.
    
    Args:
        * n_samples: number of samples used
    
    Returns:
        * empirical mean
    """
    return tf.reduce_mean(self.sample(n_samples),axis=0)
#
  def empirical_var(self, ddof=1, n_samples=1000):
    """Empirical variance of the distribution.
    
    Args:
        * n_samples: number of samples used
                * defaults to 1000
        * ddof: degrees of freedom
                * defaults to 1
    
    Returns:
        empirical variance
    """
    samples=self.sample(n_samples)
    return tf.reduce_sum((samples-tf.reduce_mean(samples))**2,axis=0)/(n_samples-ddof)
#
  def empirical_std(self, *args, **kwargs):
    """Empirical standard deviation of the distribution.
    
    Args:
        * *args: arguments to be passed to self.empirical_var
        * **kwargs: names arguments to be passed to self.empirical_var
    
    Returns:
        empirical standard deviation
    """
    return tf.sqrt(self.empirical_var(*args, **kwargs))
