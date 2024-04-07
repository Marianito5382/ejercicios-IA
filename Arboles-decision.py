#!/usr/bin/env python
# coding: utf-8

# # Cargamos las librerias y ademas es el archivo .y
# ##  la aplicación de árboles de decisión a la clasificación de clientes.

# In[1]:


#Cargamos bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[7]:


ruta= 'Mall_Customers-2.csv'


# In[8]:


data = pd.read_csv(ruta,index_col = 0)
data.head(5)


# In[9]:


data.rename({'Gender':'Genero','Age':'Edad','Annual Income (k$)':'Ingreso','Spending Score (1-100)':'Ponderacion'}, axis = 1, inplace = True)
data.head(5)


# In[10]:


data.groupby('Genero').size()


# In[11]:


data['Segmento'] = np.where(data.Ingreso >= 90 , 'Ingreso alto', np.where(data.Ingreso < 50, 'Ingreso bajo', 'Ingreso moderado'))
data


# In[12]:


data.groupby('Segmento')['Ingreso'].size()


# In[13]:


#Tratamiento de datos
train , test = train_test_split(data,test_size=0.4, stratify=data['Segmento'],random_state=42)
train.head(5)


# In[14]:


#Diagrama de dispersión de los atributos emparejados
sns.pairplot(train, hue='Segmento', height =2 , palette='colorblind');


# In[15]:


# separacion objetivo - explicativas de cada grupo
#grupo entrenamiento
x_train = train[['Ingreso','Ponderacion']] #train[['Genero','Edad','Ingreso','Ponderacion']]
y_train = train['Segmento']
#grupo prueba
x_test = train[['Ingreso','Ponderacion']] #train[['Genero','Edad','Ingreso','Ponderacion']]
y_test = train['Segmento']
#Mostrar los primeros valores
print(x_train.head(5))
print(y_train.head(5))


# In[16]:


#Creacion del modelo del arbol de decision
mod_dt = DecisionTreeClassifier(max_depth=3,random_state=1)
mod_dt.fit(x_train,y_train)
prediccion = mod_dt.predict(x_test)#Creacion del modelo del arbol de decision
mod_dt = DecisionTreeClassifier(max_depth=3,random_state=1)
mod_dt.fit(x_train,y_train)
prediccion = mod_dt.predict(x_test)


# In[17]:


#Modulos para metricas de eficiencia del modelo
from sklearn import metrics


# In[18]:


#Verificacion de la precision del arbol
print('La precision del arbol de decision es: {:.3f}'.format(metrics.accuracy_score(prediccion,y_test)))


# In[19]:


mod_dt.feature_names_in_


# In[20]:


#Visualizacion del arbol de decision
plt.figure(figsize=(10,8))
plot_tree(mod_dt,feature_names=mod_dt.feature_names_in_,
          class_names=['Ingreso bajo','Ingreso moderado','Ingreso alto'],filled=True);


# In[ ]:





# In[ ]:




