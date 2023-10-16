import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# Load the Iris dataset from scikit-learn
iris = datasets.load_iris()
X = iris.data
# Apply k-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
# Apply EM (Expectation-Maximization) clustering
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
em_labels = gmm.predict(X)
# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('k-Means Clustering')
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='viridis')
plt.title('EM (Expectation-Maximization) Clustering')
plt.show()