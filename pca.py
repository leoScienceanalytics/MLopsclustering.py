import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv('creditcustomersegmentation.csv')
colunas = ['BALANCE', 'PURCHASE ORDER', 'CREDIT LIMIT']

# Padronizar os dados (importante para PCA)
X_std = StandardScaler().fit_transform(X)

# Aplicar PCA com dois componentes principais
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_std)

# Visualizar os dados transformados antes da clusterização
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_std[:, 0], X_std[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Dados Originais')

plt.subplot(1, 2, 2)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Dados Após PCA')

plt.show()

# Aplicar K-Means nos dados transformados
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(principal_components)

# Visualizar os resultados da clusterização
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Resultados da Clusterização após PCA')
plt.legend()
plt.show()

