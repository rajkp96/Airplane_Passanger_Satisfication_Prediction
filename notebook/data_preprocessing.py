import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data():
    df =pd.read_csv("C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/Airplane_Passenger_Satisfication/data/DS-DATA.csv")
    print(df.head())

    #Basic Data Checks
    df.shape

    df.info()

    print(df.describe())
    print(df.columns)
    print(df.nunique())
    print(df['Satisfaction'].value_counts())

    #missing value analysis
    print(df.isnull().sum())

    #missing value treatment(Delay columns usually contain outliers.Median is robust
    df['Arrival Delay'] = df['Arrival Delay'].fillna(df['Arrival Delay'].median())

    #verify missing value
    print(df['Arrival Delay'].isnull().sum())

    #Duplicate treatment
    print(df.duplicated().sum())
    print(df.shape)

    #Remove Irrelevant Features
    df.drop(['ID'], axis=1, inplace=True)
    print(df.shape)

    print(df[df['Flight Distance'].str.contains(r'[^0-9]', na=False)])
    #Remove everything except digits
    df['Flight Distance'] = df['Flight Distance'].astype(str)
    df['Flight Distance'] = df['Flight Distance'].str.replace(r'[^0-9]','',regex=True)

    #Convert to Numeric
    df['Flight Distance'] = pd.to_numeric(df['Flight Distance'],errors='coerce')

    #Verify
    print(df['Flight Distance'].dtype)
    print(df['Flight Distance'].isnull().sum())

    return df
