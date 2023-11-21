import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from itertools import combinations
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv('creditcustomersegmentation.csv')
print(df.head())
print(df.shape[1])

colunas = df.columns #Criar variável q ue contém as colunas do dataframe 'dfnorm'
colunmslist = colunas.tolist() #Listarde forma organizada 
print('Lista de colunas do dataframe: ', colunmslist)


#Tratamento do dataset
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())
print(df.shape)
df = df.drop(['CUST_ID'], axis=1)
print(df)



#Teste de multicolinearidade
colunas = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']
print('Teste de multicolinearidade: ', df[colunas].corr()) #Resultado --------- não há evidências.

#SALDO: O saldo deixado nas contas dos clientes de cartão de crédito.
#COMPRAS: Valor das compras realizadas nas contas dos clientes do cartão de crédito.
#CREDIT_LIMIT: O limite do cartão de crédito.
#Clusterização será baseada em cima desses 3 grupos


#Gráfico das Distribuições Normais
sns.set(style='whitegrid')

fig, axs = plt.subplots(1, 3, figsize=(10, 5)) #Criando objeto da figura e dos eixos

sns.histplot(data=df, x='BALANCE', kde=True, color='blue', ax=axs[0])
sns.histplot(data=df, x='PURCHASES', kde=True, color='orange', ax=axs[1])
sns.histplot(data=df, x='CREDIT_LIMIT', kde=True, color='red', ax=axs[2])


axs[0].set_title('Balance Distribution')
axs[1].set_title('Purchases Distribution')
axs[2].set_title('Credit Limit Distribution')

fig.suptitle('Análise de Distribuição das Features de Clusterização')
plt.show()


#Métrica de precisão ----------------------- Inetria(WCSS)
def calculate_inertia(X):
    inertia_values = []

# Testar diferentes números de clusters (K) para o K-Means
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        inertia_values.append(inertia)
    print('Métrica de Precisão ------ Inertia: ',inertia_values)

# Plotar um gráfico do valor de Inertia em função do número de clusters (K)
    plt.plot(range(1, 11), inertia_values, marker='o')
    plt.title('Gráfico de Inertia em função do número de clusters (K)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.show()
    
    return inertia_values


def optimal_number_of_clusters(inertia_values):
    x1, y1 = 2, inertia_values[0]
    x2, y2 = 11, inertia_values[len(inertia_values)-1]

    distances = []
    for i in range(len(inertia_values)):
        x0 = i+2
        y0 = inertia_values[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = (((y2 - y1)**2 + (x2 - x1)**2)**0.5)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


clustering_data = df[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']] #Definindo as colunas que serão usadas como parâmetros de definição das clusters
clustering_data = np.array(clustering_data)
clustering_data = preprocessing.scale(clustering_data)
clustering_data = pd.DataFrame(clustering_data)

X = clustering_data
sum_of_squares = calculate_inertia(X)

number_optimal = optimal_number_of_clusters(sum_of_squares)
print(number_optimal)

 
for i in clustering_data.columns:#Método de redimensionamento, fazendo-se possível comparar as colunas selecionadas do DataFrame.
    MinMaxScaler(i)
    
    
kmeans = KMeans(n_clusters=number_optimal) # Definindo o número de clusters, irá variar de 0 até 4.
clusters = kmeans.fit_predict(clustering_data) #Previsão da segmentação de mercado
df["CREDIT_CARD_SEGMENTS"] = clusters
df['CREDIT_CARD_SEGMENTS'] = pd.DataFrame(df['CREDIT_CARD_SEGMENTS'])
print(df['CREDIT_CARD_SEGMENTS'])



colunas = df.columns #Criar variável que contém as colunas do dataframe 'dfnorm'
colunmslist = colunas.tolist() #Listar de forma organizada 
print('Lista de colunas do dataframe: ', colunmslist)

print(df)
print(df.columns)


#Métrica de precisão ------------------------ Silhuette Score
x = df["CREDIT_CARD_SEGMENTS"].values.reshape(-1, 1)
labels = kmeans.labels_

silhouette_avg = silhouette_score(x, labels)
print('Métrica  de Precisão ------- Silhouette Score: ',silhouette_avg)



#Métric de precisão ------------------- Davies Bouldin Index
kmeans = KMeans(n_clusters=number_optimal)
kmeans.fit(X)

# Os centróides dos clusters são acessados usando 'cluster_centers_' após o ajuste.
cluster_centers = kmeans.cluster_centers_




n_clusters = len(cluster_centers)

dbi = 0.0
for i in range(n_clusters):
    max_dissimilarity = 0
    for j in range(n_clusters):
        if i != j:
            dist = pairwise_distances([cluster_centers[i]], [cluster_centers[j]])[0][0]
            if dist > max_dissimilarity:
                max_dissimilarity = dist
    avg_intra_cluster_distance = np.mean(pairwise_distances(X[labels == i], [cluster_centers[i]]))
    dbi += (avg_intra_cluster_distance + avg_intra_cluster_distance) / max_dissimilarity

dbi /= n_clusters

print("DBI:", dbi)


#Métrica de precisão --------------- Calinski Harabasz Index

ch_score = calinski_harabasz_score(x, labels)
print("Índice de Calinski-Harabasz:", ch_score)




# Calcular o ARI ------------------- Não possuimos true_labels, somente predict_labels. Portanto, Não a métrica de precisão não será usada



df["CREDIT_CARD_SEGMENTS"] = df["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1: 
    "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5", 5: "Cluster 6"}) #Mapeando as clusters

print(df["CREDIT_CARD_SEGMENTS"].head(10)) #Mostrar as clusters dos 10 primeiros usuários.
print(df['CREDIT_CARD_SEGMENTS'].value_counts())
print(df['CREDIT_CARD_SEGMENTS'].value_counts().sum())

#Visualização via Web
PLOT = go.Figure()
for i in list(df["CREDIT_CARD_SEGMENTS"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = df[df["CREDIT_CARD_SEGMENTS"]== i]['BALANCE'],
                                y = df[df["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z = df[df["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'BALANCE', titlefont_color = 'black'),
                                yaxis=dict(title = 'PURCHASES', titlefont_color = 'black'),
                                zaxis=dict(title = 'CREDIT_LIMIT', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))
PLOT.show()



#Gráfico no idle do python
unique_segments = df['CREDIT_CARD_SEGMENTS'].unique() #mostrar os segmentos de 0 até 4

# Crie uma figura 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Defina cores diferentes para cada segmento
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, segment in enumerate(unique_segments):
    segment_data = df[df['CREDIT_CARD_SEGMENTS'] == segment]
    x = segment_data['BALANCE']
    y = segment_data['PURCHASES']
    z = segment_data['CREDIT_LIMIT']
    label = str(segment)
    color = colors[i % len(colors)]
    
    # Plote os pontos 3D
    ax.scatter(x, y, z, label=label, c=color, s=60)

# Defina rótulos dos eixos e legenda
ax.set_xlabel('BALANCE')
ax.set_ylabel('PURCHASES')
ax.set_zlabel('CREDIT_LIMIT')
ax.set_title('3D Scatter Plot of CREDIT_CARD_SEGMENTS')
ax.legend()

# Exiba o gráfico
plt.show()
