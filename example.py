import tensorflow as tf
import numpy as np
from truncatedDistribution import TruncatedDistribution as TD

tf.InteractiveSession()
concentration=np.array([40.],dtype=np.float32)
rate=np.array([4.],dtype=np.float32)
gamma=tf.distributions.Gamma(concentration,rate)
left=np.array([9.],dtype=np.float32)
right=30.
td=TD(gamma,left,right)
samples=td.sample(1000).eval()

import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
f,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(samples)
ax1.set_xlim(left=3,right=16)
ax1.set_title("$\Gamma(40,4)$ truncated at $9$.")
ax2.hist(gamma.sample(1000).eval())
ax2.set_xlim(left=3,right=16)
ax2.set_title("$\Gamma(40,4)$")
plt.show()