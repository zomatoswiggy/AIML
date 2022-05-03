import numpy as np
import seaborn as sns
import random

# normal distribution 
x = np.random.normal(loc= 500.0, scale=1.0, size=10000)

np.mean(x)

sample_mean = []

# Bootstrap Sampling
for i in range(40):
  y = random.sample(x.tolist(), 5)
  avg = np.mean(y)

  sample_mean.append(avg)

np.mean(sample_mean)