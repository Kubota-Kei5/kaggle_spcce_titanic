#勾配ブースティング決定木（Gradient Boosting Decision Tree）
#train score = 0.802690
# %%
import pandas as pd
import matplotlib.pyplot as mb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
train=pd.read_csv('../Data/train.csv')
test=pd.read_csv('../Data/test.csv')
# %%
x=train[['Pclass','SibSp','Parch','Sex','Fare']]
y=train[['Survived']]

x=pd.get_dummies(x, dtype=int, columns=['Pclass', 'Sex'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
# %%
from sklearn.ensemble import GradientBoostingClassifier

gbdt=GradientBoostingClassifier()
gbdt.fit(x_train_scaled, y_train)

gbdt.score(x_test_scaled, y_test)
# %%
x_for_submit = test[['Pclass', 'SibSp', 'Parch', 'Sex', 'Fare']]
submit = test[['PassengerId']]

x_for_submit=pd.get_dummies(x_for_submit, columns=['Pclass', 'Sex'])

x_for_submit['Fare']=x_for_submit['Fare'].fillna(x_for_submit['Fare'].mean())

#scaler.fitはtrainに対してfitさせたものをtestに使うのがポイント
#ただし、今回の場合はtestにfitさせたものでpredictしたほうがスコアは良かった
scaler=StandardScaler()
scaler.fit(x_train)

x_for_submit_scaled=scaler.transform(x_for_submit)

submit['Survived']=gbdt.predict(x_for_submit_scaled)
submit
# %%
submit.to_csv('../submission/submit03_gbdt.csv', index=False)
# %%
