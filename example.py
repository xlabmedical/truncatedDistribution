import tensorflow as tf
import numpy as np
from truncatedDistribution import TruncatedDistribution as TD

tf.InteractiveSession()
concentration=40.
rate=4.
gamma=tf.distributions.Gamma(concentration,rate)
left=9.
right=30.
tG=TD(gamma,left,right)
samples=tG.sample(1000).eval()
samples_org=gamma.sample(1000).eval()

import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
f,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(samples)
ax1.set_xlim(left=3,right=16)
ax1.set_title("$\Gamma(40,4)$ truncated at $9$.")
ax2.hist(samples_org)
ax2.set_xlim(left=3,right=16)
ax2.set_title("$\Gamma(40,4)$")
plt.show()

print(tG.mean().eval())
print(tG.variance().eval())

print(gamma.mean().eval())
print(gamma.variance().eval())

a=2.
b=5.
beta=tf.distributions.Beta(a,b)
left=0.35
right=1.
tB=TD(beta,left,right)
X=np.linspace(0,1,100,dtype=np.float32)
Y1=tB.cdf(X).eval()
Y2=beta.cdf(X).eval()


f,(ax1,ax2)=plt.subplots(1,2)
ax1.plot(X,Y1)
ax1.set_xlim(left=0,right=1)
ax1.set_title("$Beta(2,5)$ truncated at $0.35$.")
ax2.plot(X,Y2)
ax2.set_xlim(left=0,right=1)
ax2.set_title("$Beta(2,5)$")
plt.show()

print(tB.mean().eval())
print(tB.variance().eval())

print(beta.mean().eval())
print(beta.variance().eval())