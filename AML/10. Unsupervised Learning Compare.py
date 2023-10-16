import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

iris = datasets.load_iris()
X = iris.data
true_labels = iris.target

n_clusters = len(np.unique(true_labels))

kmeans = KMeans(n_clusters=n_clusters)
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
gmm = GaussianMixture(n_components=n_clusters)


kmeans_labels = kmeans.fit_predict(X)
hierarchical_labels = hierarchical.fit_predict(X)
gmm.fit(X)
gmm_labels = gmm.predict(X)


kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
hierarchical_ari = adjusted_rand_score(true_labels, hierarchical_labels)
gmm_ari = adjusted_rand_score(true_labels, gmm_labels)


labels = ['K-Means', 'Hierarchical', 'GMM']
ari_scores = [kmeans_ari, hierarchical_ari, gmm_ari]

plt.bar(labels, ari_scores, color=['blue', 'green', 'red'])
plt.ylabel('ARI Score')
plt.title('Clustering Algorithm Comparison')
plt.show()
