import numpy as np
import matplotlib.pyplot as plt

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        
        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.class_priors[i] = X_cls.shape[0] / X.shape[0]
            self.means[i, :] = np.mean(X_cls, axis=0)
            self.variances[i, :] = np.var(X_cls, axis=0) + 1e-9
    
    def predict(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes)))
        
        for i in range(len(self.classes)):
            prior = np.log(self.class_priors[i])
            diff = X - self.means[i, :]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances[i]) + 
                (diff ** 2) / self.variances[i], 
                axis=1
            )
            log_probs[:, i] = prior + log_likelihood
        
        return self.classes[np.argmax(log_probs, axis=1)]
    
    def predict_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes)))
        
        for i in range(len(self.classes)):
            prior = np.log(self.class_priors[i])
            diff = X - self.means[i, :]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances[i]) + 
                (diff ** 2) / self.variances[i], 
                axis=1
            )
            log_probs[:, i] = prior + log_likelihood
        
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        return exp_log / np.sum(exp_log, axis=1, keepdims=True)

np.random.seed(42)
n_samples = 150

mean0 = [1, 1]
cov0 = [[1, 0.3], [0.3, 1]]
X0 = np.random.multivariate_normal(mean0, cov0, n_samples//2)

mean1 = [4, 4]
cov1 = [[1, -0.2], [-0.2, 1]]
X1 = np.random.multivariate_normal(mean1, cov1, n_samples//2)

X = np.vstack((X0, X1))
y = np.array([0]*len(X0) + [1]*len(X1))

indices = np.random.permutation(n_samples)
train_size = int(n_samples * 0.8)
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = GaussianNaiveBayes()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.2f}")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(10, 8))

contour = plt.contourf(xx, yy, probs, levels=25, cmap="coolwarm", alpha=0.6)
plt.colorbar(contour, label="Probability of Class 1")


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
            cmap="bwr", edgecolors='k', s=50, label="Train Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
            cmap="bwr", edgecolors='k', s=100, marker='s', label="Test Data")


y_test_pred = model.predict(X_test)
misclassified = np.where(y_test_pred != y_test)
plt.scatter(X_test[misclassified, 0], X_test[misclassified, 1],
            s=150, edgecolors='k', facecolors='none', linewidths=2, 
            marker='o', label="Misclassified")


plt.title("Gaussian Naive Bayes Classifier", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend(loc="best")
plt.grid(alpha=0.2)

class0_ratio = np.sum(y == 0) / len(y)
class1_ratio = np.sum(y == 1) / len(y)
plt.figtext(0.15, 0.02, 
            f"Class Distribution: Class 0 = {class0_ratio:.2f}, Class 1 = {class1_ratio:.2f}",
            fontsize=10, ha="left")

plt.tight_layout()
plt.show()
