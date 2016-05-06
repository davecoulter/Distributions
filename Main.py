import copy
from math import *
from random import *
import numpy as np
import matplotlib.pyplot as plt

# Find the marginal probability bin which contains the candidate
def FindIndex(bins, candidate):
    for index in range(len(bins)):
        lower = index if index == 0 else bins[index - 1]
        upper = bins[index]
        if lower < candidate <= upper:
            return index

# The distribution to use... Swap this out, but make sure it is normalizable
def GaussianDistribution(input, **kwargs):
    return exp(-pow(input - kwargs["mean"], 2)/(2 * pow(kwargs["std_dev"], 2)))


# Set up PDF resolution
start = 0
end = 10
n = 1000 # resolution
del_t = (end - start) / float(n) # step size
steps = np.arange(start, end, del_t) # step array

# Hold computed values, sized for the number of steps
weights = [0] * n
marginal_prob = [0] * n
frequency = [0] * n

# Number of times to randomly sample the PDF
samples = 10000

# Build normally distributed weights
i = 0
for t in steps:
    weights[i] = GaussianDistribution(t, mean=5, std_dev=0.75) * del_t
    i += 1

# Normalize
A = sum(weights)
weights[:] = [x / A for x in weights]
model = copy.deepcopy(weights)

# Create a model by scaling the PDF by the number of samples &
# Build marginal probability array
for i in range(n):
    marginal_prob[i] = sum(weights[:i + 1])
    model[i] *= samples

# Build random sampling of PDF
for i in range(samples):
    index = FindIndex(marginal_prob, random())
    frequency[index] += 1

# Plot random numbers subject to the PDF, and the model PDF
plt.plot(steps, frequency, 'bx', steps, model, 'r-')
plt.show()
