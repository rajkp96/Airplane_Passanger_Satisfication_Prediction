import pandas as pd

from train_model import train_models
from sklearn.metrics import accuracy_score, classification_report

(lr,dt,rf,xgb,pred_lr,pred_dt,pred_rf,pred_xgb,X_test,X_test_scaled,y_test,feature_names) = train_models()

#Logistic Regression Model
print("========== Logistic Regression ==========")
print("Accuracy:",accuracy_score(y_test,pred_lr))
print(classification_report(y_test,pred_lr))

#Model-2 Decision Tree
print("========== Decision Tree ==========")
print("Accuracy:",accuracy_score(y_test,pred_dt))
print(classification_report(y_test,pred_dt))

#Model-3 Random Forest
print("========== Random Forest ==========")
print("Accuracy:",accuracy_score(y_test,pred_rf))
print(classification_report(y_test,pred_rf))

#Model -4 XGBoost
print("========== XGBoost ==========")
print("Accuracy:",accuracy_score(y_test,pred_xgb))
print(classification_report(y_test,pred_xgb))

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
y_prob = xgb.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

results = pd.DataFrame({'Model':['Logistic Regression','Decision Tree','Random Forest','XGBoost'],
'Accuracy':[accuracy_score(y_test,pred_lr), accuracy_score(y_test,pred_dt), accuracy_score(y_test,pred_rf), accuracy_score(y_test,pred_xgb)]})
print(results.sort_values(by='Accuracy',ascending=False))

#Feature Importance from XGBoost:

importance = pd.DataFrame({"Feature": feature_names,"Importance": xgb.feature_importances_})
importance = importance.sort_values(by='Importance',ascending=False)
print(importance.head(20))
