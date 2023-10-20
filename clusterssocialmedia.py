#Bibliotecas 
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pandas as pd
pio.templates.default = "plotly_white"
from sklearn.preprocessing import MinMaxScaler


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
z = df['Ratings']
x = df["Average Screen Time"]
y = df["Average Spent on App (INR)"]
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
z = df['Ratings']
x = df["Average Screen Time"]
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
    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(clustering_data)
df["Segments"] = clusters
print(df['Segments'].head(10))

print(df["Segments"].value_counts())

df["Segments"] = df["Segments"].map({0: "Retained", 1: 
    "Churn", 2: "Needs Attention"})
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