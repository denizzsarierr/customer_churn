import pandas as pd
from preprocces import preproccess
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV
import numpy as np

telco_data_base = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

telco_data = preproccess(telco_data_base)

print(telco_data.head())
print(telco_data.info())

y = telco_data["Churn"]
X = telco_data.drop("Churn",axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# RESAMPLE MODEL , FIXING IMBALANCE
sm = SMOTEENN(random_state=42)
X_res ,y_res = sm.fit_resample(X_train,y_train)


#X_res,y_res = sm.fit_resample(X_train,y_train)

# LOGISTIC REGRESSION MODEL

lr = LogisticRegression(class_weight="balanced",random_state=42,max_iter=2500)
lr.fit(X_res,y_res)

y_predict_lr = lr.predict(X_test)

print("=========== LR MODEL ===========")
print(classification_report(y_test,y_predict_lr))
print("=============CONFUSION MATRIX=============")
print(confusion_matrix(y_test,y_predict_lr))
print("=============ROC-AUC SCORE=====")
y_prob = lr.predict_proba(X_test)
y_prob = y_prob[:,1]

print(roc_auc_score(y_test,y_prob))

print("===========================================")



rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)

rf.fit(X_res,y_res)

y_predict_rf = rf.predict(X_test)

print("=========== RF MODEL ===========")
print(classification_report(y_test,y_predict_rf))

print("=============CONFUSION MATRIX=============")
print(confusion_matrix(y_test,y_predict_rf))

print("=============ROC-AUC SCORE=====")
y_prob1 = rf.predict_proba(X_test)
y_prob1 = y_prob1[:,1]
print(roc_auc_score(y_test,y_prob1))

print("==========================================")


dt = DecisionTreeClassifier(criterion = "gini",random_state = 42,max_depth=6, min_samples_leaf=8)

dt.fit(X_res,y_res)

y_predict_dt = dt.predict(X_test)

print("=========== DT MODEL ===========")
print(classification_report(y_test,y_predict_dt))

print("=============CONFUSION MATRIX=============")
print(confusion_matrix(y_test,y_predict_dt))

print("=============ROC-AUC SCORE=====")
y_prob2 = rf.predict_proba(X_test)
y_prob2 = y_prob2[:,1]
print(roc_auc_score(y_test,y_prob2))

print("==========================================")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = cross_val_score(xgb, X_res, y_res, cv=kf, scoring='f1')
print("CV F1 Scores:", cv_f1_scores)
print("CV Mean F1 Score:", np.mean(cv_f1_scores))


xgb.fit(X_res, y_res)


y_pred_test = xgb.predict(X_test)
y_prob_test = xgb.predict_proba(X_test)[:,1]

print("=========== XGBoost on Test Set ===========")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_test))