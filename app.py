import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


mnist = datasets.fetch_openml('mnist_784')
X = np.array(mnist.data)
y = np.array(mnist.target, dtype='int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(gamma='scale', kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def plot_digits(images, labels, predictions, n=10):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {labels[i]}\nPred: {predictions[i]}')
        plt.axis('off')
    plt.show()

plot_digits(X_test, y_test, y_pred)