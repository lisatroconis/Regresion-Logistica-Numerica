
import pandas as pd
import numpy as np
#from sklearn import preprocessing
import matplotlib.pyplot as plt 
#plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)

"""# Preparación de la data"""

from google.colab import drive
drive.mount('/content/drive')

df_data = pd.read_excel("/content/drive/MyDrive/EAN_Documentos/Curso Python Enero 2022/2. Solución/final_df.xlsx")
display(df_data.shape)
display(df_data.head())

df_data.columns

"""# Modelo de regresión logística

Este modelo se construye con el objetivo de evaluar la probabilidad de que una variable tome determinado valor conociendo información de otras. En este notebook, vamos a evaluar la probabilidad de que un hogar sea pobre dependiendo de su nivel educativo y la cantidad de bienes que posee.
Un hogar se define pobre si su ingreso per capita es menor a $327.674

Definición de la variable pobreza
"""

df_data["pobre"]=0
df_data.loc[df_data.Ingresos<327674,"pobre"]=1
df_data['pobre'].describe()

df_data["pobre"].value_counts()

sns.countplot(x="pobre",data=df_data)
plt.show()
#plt.savefig("count_plot")

f, axs = plt.subplots(1, 2)
sns.boxplot(data=df_data["Nivel_Educativo"], ax=axs[0])
sns.boxplot(data=df_data["Bienes"])

"""## Medidas estadísticas por grupo"""

display(df_data[["Nivel_Educativo","Bienes","pobre"]].groupby("pobre").describe().T)

"""## Distribución de la pobreza por Nivel educativo"""

sns.kdeplot(data=df_data,x="Nivel_Educativo",hue="pobre")

"""## Distribución de la pobreza por cantidad de bienes"""

sns.kdeplot(data=df_data,x="Bienes",hue="pobre")

"""# Estimación del modelo"""

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X=df_data[["Nivel_Educativo","Bienes"]]
y=df_data["pobre"]

display(X.head())
display(y.head())

X = sm.add_constant(X)

display(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

display(X_train.head())
display(y_train.head())

logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

X_test.head()

y_test.head()

Resultados = pd.DataFrame({"Probabilidades":result.predict(X_test)})

Resultados.head()

Resultados.describe()

Resultados["Predicciones"]=1
Resultados.loc[Resultados.Probabilidades<0.400168,"Predicciones"]=0

Resultados.describe()

"""## Medidas de validación"""

## Matriz de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, Resultados.Predicciones)
print(confusion_matrix)

"""**Accuracy:** Porcentaje de aciertos

**Precisión:** Porcentaje de predichos que son reales.

**Recall**: Porcentaje de reales predichos.

**F1-score:**: Combinación de la precisión y el recall

"""

confusion_matrix[1] # Metricas de rendimiento

Accuracy= (confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0]+confusion_matrix[0][1]+
                                                           confusion_matrix[1][0]+confusion_matrix[1][1])
display(Accuracy)

Precision_0=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]) 
display(Precision_0)

Recall_0=confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]) #TPR - RAZON DE VERDADEROS POSITIVOS. 
display(Recall_0)

F1_score_0=2/((1/Precision_0)+(1/Recall_0))
display(F1_score_0)

from sklearn.metrics import classification_report
print(classification_report(y_test, Resultados.Predicciones))

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, Resultados.Predicciones)
fpr, tpr, thresholds = roc_curve(y_test, result.predict(X_test))
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()