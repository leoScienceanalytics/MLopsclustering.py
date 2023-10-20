import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go


df = pd.read_csv('creditcustomersegmentation.csv')
print(df.head())
print(df.shape[1])

colunas = df.columns #Criar variável que contém as colunas do dataframe 'dfnorm'
colunmslist = colunas.tolist() #Listar de forma organizada 
print('Lista de colunas do dataframe: ', colunmslist)

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

#SALDO: O saldo deixado nas contas dos clientes de cartão de crédito.
#COMPRAS: Valor das compras realizadas nas contas dos clientes do cartão de crédito.
#CREDIT_LIMIT: O limite do cartão de crédito.
#Clusterização será baseada em cima desses 3 grupos

clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]] #Definindo as colunas que serão usadas como parâmetros de definição das clusters
for i in clustering_data.columns:#Método de redimensionamento, fazendo-se possível comparar as colunas selecionadas do DataFrame.
    MinMaxScaler(i)
    
    
kmeans = KMeans(n_clusters=5) # Definindo o número de clusters, irá variar de 0 até 4.
clusters = kmeans.fit_predict(clustering_data) #Previsão da segmentação de mercado
df["CREDIT_CARD_SEGMENTS"] = clusters

df["CREDIT_CARD_SEGMENTS"] = df["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1: 
    "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"}) #Mapeando as clusters

print(df["CREDIT_CARD_SEGMENTS"].head(10)) #Mostrar as clusters dos 10 primeiros usuários.


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
