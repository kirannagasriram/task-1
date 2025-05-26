import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"C:\Users\kiran\OneDrive\Desktop\Titanic-Dataset.csv")
print(df.info())
print(df.head())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.dropna(inplace=True)

print("\nAfter handling missing values:")
print(df.info())

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nNormalized numerical features:")
print(df[numerical_cols].describe().T)

plt.figure(figsize=(12,4))
for i, col in enumerate(['Age','Fare']):
    plt.subplot(1,2,i+1)
    plt.boxplot(df[col])
    plt.title(f'{col} distribution')
plt.tight_layout()
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
mask = (df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)
df_clean = df[mask]

print("\nBefore removing outliers:", df.shape)
print("After removing outliers:", df_clean.shape)

print("\nCleaned DataFrame sample:")
print(df_clean.head())
