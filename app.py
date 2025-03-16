import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


mnist = datasets.fetch_openml('mnist_784')
X = np.array(mnist.data)
y = np.array(mnist.target, dtype='int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)