import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 100)  

class KNNRegressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sum(np.abs(self.X_train - x), axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_values = self.y_train[k_indices]
            predictions.append(np.mean(k_values))
        return np.array(predictions)

model = KNNRegressor(k=5)
model.fit(X_train, y_train)

X_test = np.linspace(0, 10, 500).reshape(-1, 1)
y_pred = model.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='lightblue', label='Training data', alpha=0.6)
plt.plot(X_test, y_pred, color='red', label='KNN Prediction (k=5)')
plt.scatter(X_test, y_pred, color='green', s=10, label='Predictions')  # Show prediction points
plt.title("k-NN Regression with Manhattan Distance")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
