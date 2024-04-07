#!/usr/bin/env python
# coding: utf-8

# # Cargamos librerias
# ## aplicación de la segmentación por K-means aplicada a la clasificación de especies para flores Iris utilizando el conjunto de datos provisto por los paquetes de python.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mpl_toolkits import mplot3d


# In[2]:


data = sns.load_dataset('iris')
data.head(5)


# In[3]:


data.describe().T


# In[4]:


data.sepal_length.plot.hist();


# In[5]:


sns.set()


# In[6]:


data.sepal_length.plot.hist()
plt.xlabel('Ancho de sepalo');


# In[7]:


data.groupby('species')['sepal_length'].describe()


# In[8]:


scaler = StandardScaler()
col_a_escalar = ['sepal_length', 'petal_length', 'petal_width']


# In[9]:


datos_a_escalar = data.copy()


# In[10]:


datos_a_escalar[col_a_escalar] = scaler.fit_transform(data[col_a_escalar])


# In[11]:


data.head(10)


# In[12]:


datos_a_escalar.head(10)


# In[13]:


data.sepal_length.plot.hist()
plt.xlabel('ancho de sepalo');


# In[14]:


datos_a_escalar.sepal_length.plot.hist()
plt.xlabel('ancho de sepalo estandarizado');


# In[15]:


cluster_cols = ['sepal_length', 'petal_length','petal_width']
datos_a_escalar[cluster_cols].head()


# In[16]:


modelo2 = KMeans(n_clusters=3, random_state=42)
modelo2.fit(datos_a_escalar[cluster_cols])


# In[17]:


datos_a_escalar['Cluster2']= modelo2.predict(datos_a_escalar[cluster_cols])
datos_a_escalar


# In[18]:


from sklearn import decomposition


# In[19]:


pca=decomposition.PCA(n_components=2)
pca_res = pca.fit_transform(datos_a_escalar[cluster_cols])


# In[20]:


pca_res


# In[21]:


datos_a_escalar['pc1'] = pca_res[:,0]
datos_a_escalar['pc2'] = pca_res[:,1]


# In[22]:


datos_a_escalar


# In[ ]:





# In[23]:


marcador = ['x', '*', '.', '|', '_']
for segmento in range(3):
    temp = datos_a_escalar[datos_a_escalar.Cluster2 == segmento]
    plt.scatter(temp.pc1, temp.pc2, marker = marcador[segmento], label = 'Cluster '+ str(segmento))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend();


# In[24]:


data['Segmento IA'] = datos_a_escalar.Cluster2
data


# In[25]:


data.groupby('Segmento IA')[col_a_escalar].mean()


# In[26]:


data.groupby('Segmento IA')[col_a_escalar].mean().plot.bar();


# In[27]:


col_segmento = ['sepal_length', 'petal_length']
X = datos_a_escalar[col_segmento]


# In[28]:


puntuacionInercia = []
for k in range(2, 11):
    inercia = KMeans(n_clusters=k, random_state=42).fit(X).inertia_
    puntuacionInercia.append(inercia)
puntuacionInercia


# In[29]:


tipos = data[['sepal_length', 'petal_length', 'petal_width']].copy()


# In[30]:


# Creacion de modelo
km = KMeans(n_clusters=3, n_init=100, max_iter=1000, init='random')


# In[31]:


prediccionkm = km.fit_predict(tipos)


# In[32]:


calinski_harabasz_score(tipos, prediccionkm)
#Mientras mas alto sea el punteo, es más denso y distanciado


# In[33]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[34]:


fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.scatter3D(tipos['sepal_length'], tipos['petal_length'], tipos['petal_width'], c=prediccionkm, cmap='tab10')
plt.title('Segmentacion')
ax.set_xlabel('Longitud de sepalo')
ax.set_ylabel('Longitdu de petalo')
ax.set_zlabel('Ancho de petalo')

plt.plot([], [], color='b', label = 'Setosa')
plt.plot([], [], color='g', label = 'Versicolor')
plt.plot([], [], color='r', label = 'Virginica')

plt.show()
plt.legend();


# In[35]:


data.groupby('species')['sepal_length'].describe()


# In[ ]:




