import numpy as np
from sklearn import datasets


mnist = datasets.fetch_openml('mnist_784')
X = np.array(mnist.data)
y = np.array(mnist.target, dtype='int')