import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('parkinsons.data')
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)

#Разделим на тестовую и обучающую выборку. Выборки делить в соотношении 80% обучающая, 20% - тестовая.
atrain, atest, btrain, btest = train_test_split(X, labels, test_size=0.2)

#Создадим модель классификатора и обучим его на данных обучающей выборки
model = XGBClassifier()
model.fit(atrain, btrain)

#Оценка результата
res = [round(yhat) for yhat in model.predict(atest)]
print(f"{metrics.accuracy_score(btest, res):.2%}")

#Матрица ошибок
ax = sns.heatmap(cf_matrix/ cf_matrix.sum(axis=1, keepdims=True), annot=True, cmap='Blues', fmt='.4f', square=True)

ax.set_title('Матрица ошибок\n\n');
ax.set_xlabel('\nПредсказанные метки')
ax.set_ylabel('Истинные метки')

plt.show()
