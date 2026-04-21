import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("Crop_recommendation.csv")
print(df.head())

print(df.tail())
print(df.info())
print(df.describe())
print(df['label'].value_counts())

map1 = df.hist(bins=50,figsize=(12,8))
plt.show()