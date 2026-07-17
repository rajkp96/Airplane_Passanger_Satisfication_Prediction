import xgboost

def train_models():

    from feature_engineering import prepare_features

    (X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled,scaler,feature_names) = prepare_features()

    #Model 1-Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)

    #Model-2 Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=10,random_state=42)
    dt.fit(X_train,y_train)
    pred_dt = dt.predict(X_test)

    #Model-3 Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train,y_train)
    pred_rf = rf.predict(X_test)

    #Model -4 XGBoost
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42)
    xgb.fit(X_train,y_train)
    pred_xgb = xgb.predict(X_test)


    import joblib
    print(joblib.dump(xgb,"C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/Airplane_Passenger_Satisfication/model/Airline Passenger Satisfaction Prediction_model.pkl"))
    print(joblib.dump(scaler,"C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/Airplane_Passenger_Satisfication/notebook/final_scaler.pkl"))

    return (lr,dt,rf,xgb,pred_lr,pred_dt,pred_rf,pred_xgb,X_test,X_test_scaled,y_test,feature_names)

