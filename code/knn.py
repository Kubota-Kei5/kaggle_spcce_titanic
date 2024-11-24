#k-近傍法（k-nearest neighbor）
#train score = 
# %%
import pandas as pd
import matplotlib.pyplot as mb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
train=pd.read_csv('../data/train.csv')
test=pd.read_csv('../data/test.csv')
# %%
train.head()
# %%
train.dtypes
# %%
train.isnull().sum()
# %%
test.dtypes
# %%
test.isnull().sum()
# %%
x=train[['HomePlanet','CryoSleep','Age','VIP']]
y=train[['Transported']]

x['HomePlanet']=x['HomePlanet'].fillna("hp_unknown")
x['CryoSleep']=x['CryoSleep'].fillna("cs_unknown")
x['Age']=x['Age'].fillna(x['Age'].mean())
x['VIP']=x['VIP'].fillna("vip_unknown")

# %%
x=pd.get_dummies(x, dtype=int, columns=['HomePlanet', 'CryoSleep','VIP'])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

x_train

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)

knn=KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)

knn.score(x_test_scaled, y_test)
# %%
x_for_submit = test[['HomePlanet','CryoSleep','Age','VIP']]
submit = test[['PassengerId']]

x_for_submit['HomePlanet']=x_for_submit['HomePlanet'].fillna("hp_unknown")
x_for_submit['CryoSleep']=x_for_submit['CryoSleep'].fillna("cs_unknown")
x_for_submit['Age']=x_for_submit['Age'].fillna(x_for_submit['Age'].mean())
x_for_submit['VIP']=x_for_submit['VIP'].fillna("vip_unknown")

x_for_submit=pd.get_dummies(x_for_submit, dtype=int, columns=['HomePlanet', 'CryoSleep','VIP'])

#scaler.fitはtrainに対してfitさせたものをtestに使うのがポイント
#ただし、今回の場合はtestにfitさせたものでpredictしたほうがスコアは良かった
scaler=StandardScaler()
scaler.fit(x_train)

x_for_submit_scaled=scaler.transform(x_for_submit)

submit['Transported']=knn.predict(x_for_submit_scaled)
submit
# %%
submit.to_csv('../submission/submit01_knn.csv', index=False)
# %%
