import pandas as pd
import numpy as np
import pickle
from flask import Flask
df = pd.read_csv('advertising_data.csv')
df['TV'].fillna(df['TV'].mean(),inplace=True)
df['radio'].fillna(0, inplace=True)
df['newspaper'].fillna(df['newspaper'].mean(), inplace=True)
x = df.iloc[:,:3]
y = df.iloc[:,-1]
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
print(reg.predict([[44.5,39.3,45.1]]))
pickle.dump(reg,open('model.pkl','wb'))
