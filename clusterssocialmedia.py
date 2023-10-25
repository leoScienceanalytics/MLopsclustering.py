#Bibliotecas 
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pandas as pd
pio.templates.default = "plotly_white"
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score    
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


#Conectando Base de dados ao ambiente
df = pd.read_csv('userbehaviour.csv')
print(df.head(20))

#Visualização Técnica
print(df.info())

#Análises iniciais
print(f'Average Screen Time = {df["Average Screen Time"].mean()}')
print(f'Highest Screen Time = {df["Average Screen Time"].max()}')
print(f'Lowest Screen Time = {df["Average Screen Time"].min()}')

print(f'Average Spend of the Users = {df["Average Spent on App (INR)"].mean()}')
print(f'Highest Spend of the Users = {df["Average Spent on App (INR)"].max()}')
print(f'Lowest Spend of the Users = {df["Average Spent on App (INR)"].min()}')


# Definindo as colunas de dados
z = df['Ratings'].values
x = df["Average Screen Time"].values
y = df["Average Spent on App (INR)"].values
size = df["Average Spent on App (INR)"]
size1= df['Ratings']
colors = df["Status"]


# Criando um dicionário para mapear categorias de "Status" em cores
color_mapping = {
    "Installed": "blue",
    "Uninstalled": "red"
}


mapped_colors = [color_mapping[status] for status in colors]


#Usuários que Desinstalaram
#Média Gasta em função do Tempo médio de tela
# Criando o gráfico de dispersão

plt.scatter(x, y, s=size, c=mapped_colors)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "k--", label="Linha de Tendência")
# Adicione um título
plt.title("Relationship Between Spending Capacity and Screentime")
# Rotule os eixos
plt.xlabel("Average Screen Time")
plt.ylabel("Average Spent on App (INR)")
# Mostre o gráfico
plt.show()

#Rating por Tempo médio de tela
#Definindo novas colunas
z = df['Ratings'].values
x = df["Average Screen Time"].values
size1= df['Ratings']
colors = df["Status"]

#Mapeando Variáveis catégoricas
color_map = {
    "Installed": "blue",
    "Uninstalled": "red"
}

mapped_colors = [color_map[status] for status in colors]

# Crie o gráfico de dispersão
plt.scatter(x, z, s=size1, c=mapped_colors)
# Adicione um título
plt.title("Relationship Between Rating and Screentime")
# Rotule os eixos
plt.xlabel("Average Screen Time")
plt.ylabel("Rating")
# Mostre o gráfico
plt.show()

clustering_data = df[["Average Screen Time", "Left Review", 
                        "Ratings", "Last Visited Minutes", 
                        "Average Spent on App (INR)", 
                        "New Password Request"]]


for i in clustering_data.columns:
    MinMaxScaler(i)


kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(clustering_data)
df["Segments"] = clusters
df["Segments"] = pd.DataFrame(df["Segments"]) #Transformando o valor em um DataFrame de verdade

print(df['Segments'].head(50))

print(df["Segments"].value_counts())

#Métrica de precisão ------------------------ Silhuette Score
x = df["Segments"].values.reshape(-1, 1)
labels = kmeans.labels_
silhouette_avg = silhouette_score(x, labels)
print('Métrica  de Precisão ------- Silhouette Score: ',silhouette_avg)




#Métrica de precisão ----------------------- Inetria(WCSS)

inertia_values = []

# Testar diferentes números de clusters (K) para o K-Means
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df["Segments"].values.reshape(-1, 1))
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
print('Métrica de Precisão ------ Inertia: ',inertia_values)

# Plotar um gráfico do valor de Inertia em função do número de clusters (K)
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Gráfico de Inertia em função do número de clusters (K)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.show()


#Métric de precisão ------------------- Davies Bouldin Index
# Crie um modelo de clustering (K-Means, por exemplo) com o número de clusters desejado.
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(x)

# Obtenha os rótulos de cluster para cada ponto de dados.


# Calcule o Índice Davies-Bouldin para avaliar a qualidade dos clusters.
db_score = davies_bouldin_score(x, kmeans.labels_)

print(f"Índice Davies-Bouldin: {db_score}")




#Métrica de precisão --------------- Calinski Harabasz Index
ch_score = calinski_harabasz_score(x, labels)
print('Métrica de precisão ------------- Calinks Harabasz Index: ', ch_score) #Valor  de 1; analisar e comparar com modelos com n_cluster diferentes



#Métrica de precisão ---------------ARI(Adjusted Rand Score)


# Calcular o ARI ------------------- Não possuimos true_labels, somente predict_labels. Portanto, Não a métrica de precisão não será usada


df["Segments"] = df["Segments"].map({0: "Retained", 1: 
    "Churn", 2: "Needs Attention"})

print(df['Segments'].head(50))

print(df["Segments"].value_counts())

PLOT = go.Figure()
for i in list(df["Segments"].unique()):
    

    PLOT.add_trace(go.Scatter(x = df[df["Segments"]== i]['Last Visited Minutes'],
                                y = df[df["Segments"] == i]['Average Spent on App (INR)'],
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='Last Visited Minutes: %{x} <br>Average Spent on App (INR): %{y}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   yaxis_title = 'Average Spent on App (INR)',
                   xaxis_title = 'Last Visited Minutes',
)
PLOT.show()