import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from data_preprocessing import preprocess_data
df = preprocess_data()

print(df.head())

print(df.describe())


#Categorical Columns
cat_cols = df.select_dtypes(include='object').columns
print(cat_cols)

#Countplot for All Categorical Features
for col in cat_cols:

    plt.figure(figsize=(8,4))
    sns.countplot(x=df[col])
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


#Numerical Features
num_cols = df.select_dtypes(include=['int64','float64']).columns
print(num_cols)

#Univariate Analysis Numerical Features
for col in num_cols: 
    plt.figure(figsize=(6,3))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

#Detect outliers Boxplot analysis
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()

#Satisfaction vs Gender
plt.figure(figsize=(8,5))
sns.countplot(x='Gender',hue='Satisfaction',data=df)
plt.title('Satisfaction by Gender')
plt.show()

#Satisfaction vs Type of Travel
plt.figure(figsize=(8,5))
sns.countplot(x='Type of Travel',hue='Satisfaction',data=df)
plt.title('Satisfaction by Travel Type')
plt.show()

#Satisfaction vs Class
plt.figure(figsize=(8,5))
sns.countplot(x='Class',hue='Satisfaction',data=df)
plt.title('Satisfaction by Class')
plt.show()

#Satisfaction vs Age
plt.figure(figsize=(10,5))
sns.boxplot(x='Satisfaction',y='Age',data=df)
plt.show()

#Flight Distance vs satisfication
plt.figure(figsize=(10,5))
sns.boxplot(x='Satisfaction',y='Flight Distance',data=df)
plt.title("Flight Distance vs Satisfaction")
plt.show()

#Correlation Heatmap
from sklearn.preprocessing import LabelEncoder
corr_df = df.copy()
le = LabelEncoder()
for col in corr_df.select_dtypes(include='object').columns:
    corr_df[col] = le.fit_transform(corr_df[col])

plt.figure(figsize=(18,10))
sns.heatmap(corr_df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()