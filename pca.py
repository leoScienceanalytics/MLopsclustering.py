import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Padronizar os dados (importante para PCA)
X_std = StandardScaler().fit_transform(X)

# Aplicar PCA com dois componentes principais
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_std)

# Criar um DataFrame com os componentes principais
components_df = pd.DataFrame(data=principal_components, columns=['Componente 1', 'Componente 2'])
final_df = pd.concat([components_df, pd.Series(y, name='Target')], axis=1)

# Visualizar os dados transformados
plt.figure(figsize=(8, 6))
targets = [0, 1, 2]
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indices_to_keep = final_df['Target'] == target
    plt.scatter(final_df.loc[indices_to_keep, 'Componente 1'],
                final_df.loc[indices_to_keep, 'Componente 2'],
                c=color, s=50)

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(targets, title='Classes')
plt.show()
