import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data

def prepare_features():
    df = preprocess_data()

    #Outlier Detention
    for col in ['Departure Delay', 'Arrival Delay', 'Flight Distance']:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) |(df[col] > upper)]

        print(f"\n{col}")
        print("Outliers:", outliers.shape[0])
        print("Percentage:",round((outliers.shape[0]/len(df))*100,2), "%")

    #we need to cap 
    for col in ['Departure Delay', 'Arrival Delay','Flight Distance']:
    
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
    
        IQR = Q3 - Q1
    
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    
        print(f"\n{col}")
        print("Lower Bound:", lower)
        print("Upper Bound:", upper)

    #we need to cap these
    dep_cap = df['Departure Delay'].quantile(0.99)
    arr_cap = df['Arrival Delay'].quantile(0.99)
    fd_cap = df['Flight Distance'].quantile(0.99)

    print("Departure Delay 99th Percentile:", dep_cap)
    print("Arrival Delay 99th Percentile:", arr_cap)
    print("Flight Distance 99th Percentile:", fd_cap)

    df['Departure Delay'] = np.where(df['Departure Delay'] > dep_cap,dep_cap,df['Departure Delay'])
    df['Arrival Delay'] = np.where(df['Arrival Delay'] > arr_cap,arr_cap,df['Arrival Delay'])
    df['Flight Distance'] = np.where(df['Flight Distance'] > fd_cap,fd_cap,df['Flight Distance'])

    #verify
    print("New Departure Delay Max:",
        df['Departure Delay'].max())
    print("New Arrival Delay Max:",
        df['Arrival Delay'].max())
    print("New Flight Distance:",
        df['Flight Distance'].max())

    #as per co-relation graph arrival delay and departure delay have high corelation so we have to remove one od the column)
    df.drop(['Arrival Delay'], axis=1, inplace=True)

    print(df.shape)

    #Encoding
    #Separate Features & Target
    X = df.drop('Satisfaction', axis=1)
    y = df['Satisfaction']
    feature_names= X.columns

    #One-Hot Encode Gender, Customer Type, Type of Travel
    X = pd.get_dummies(X,columns=['Gender','Customer Type','Type of Travel'],drop_first=True)

    #Encoding Class
    X['Class'] = df['Class'].map({'Economy': 0,'Economy Plus': 1,'Business': 2})

    #verify
    print(X[['Class']].head())

    #Label Encode Target Variable
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    #verify
    print(dict(zip(le_target.classes_,le_target.transform(le_target.classes_))))

    #final verification
    print(X.head())

    #converting boolean column to int
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    #final verification
    print(X.head())

    #Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

    print(X_train.shape)
    print(X_test.shape)

    #Feature scaling for logistic regression model
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled,scaler,feature_names)