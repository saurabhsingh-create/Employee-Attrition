# !pip install lightgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import lightgbm as lgbm
# !pip install shap
import shap
from lightgbm import LGBMClassifier, plot_importance, LGBMRegressor
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.metrics import f1_score, roc_auc_score, auc, classification_report,accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, r2_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import plotly
import plotly.express as px

import os
os.getcwd()

df1 = pd.read_excel(r"C:\Users\saurabhsingh\Downloads\Case Study_Attrition.xlsx", sheet_name=1)

print(df1.shape)
df1.info()
df1.Attrition.value_counts()



attFeatures = []
for i in df1.columns:
    attFeatures.append([i, df1[i].nunique(), df1[i].drop_duplicates().values])
pd.DataFrame(attFeatures, columns = ['Features', 'Unique Number', 'Values'])


# Data Processing 


#Drop columns with single value throughout
try:
    cols_drop = ['Over18','EmployeeCount','StandardHours']
    df1.drop(cols_drop, axis =1, inplace = True)
except:
    pass


#Label encode
df1.Gender = np.where(df1.Gender =="Male", 1,0)
df1.Attrition = np.where(df1.Attrition =="Yes", 1,0)

#Inpute for null values
df1 = df1.fillna(df1.mean())

ohe_columns = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
df1=pd.get_dummies(df1)
df1.head()


df2 = pd.read_excel(r"C:\Users\saurabhsingh\Downloads\Case Study_Attrition.xlsx", sheet_name=2)
df3 = pd.read_excel(r"C:\Users\saurabhsingh\Downloads\Case Study_Attrition.xlsx", sheet_name=3)

df3['EnvironmentSatisfaction'] = df3['EnvironmentSatisfaction'].fillna(df3['EnvironmentSatisfaction'].mode()[0])
df3['JobSatisfaction'] = df3['JobSatisfaction'].fillna(df3['JobSatisfaction'].mode()[0])
df3['WorkLifeBalance'] = df3['WorkLifeBalance'].fillna(df3['WorkLifeBalance'].mode()[0])
df3.info()

df3.head()

df1 = df1.merge(df2, on = 'EmployeeID', how = 'left', validate = "1:1")
df1 = df1.merge(df3, on = 'EmployeeID', how = 'left', validate = "1:1")
df1.head()




f,ax = plt.subplots(figsize=(40,40))
sns.heatmap(df1.corr(), annot =True, linewidth =".5", fmt =".2f", cmap='coolwarm')
plt.show()



df1.info()


sns.countplot(x=df1.Attrition, data= df1)
plt.show()
print(df1.Attrition.value_counts())
df.info()


df1.hist(bins=50, figsize=(25,15))
plt.show()


y = df1['Attrition']
X = df1.drop(columns = ['Attrition', 'EmployeeID'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


print(y_train.value_counts()/X_train.shape[0])
print(y_test.value_counts()/X_test.shape[0])


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.43, solver="lbfgs", max_iter=200)
classifier.fit(X_train, y_train)


preds = classifier.predict(X_test)


cm = confusion_matrix(y_test, preds)
score = roc_auc_score(y_test, preds)
print(score)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(cm)


from sklearn.model_selection import GridSearchCV
grid = {"C": np.arange(0.2,0.5,0.01),
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [200]
       }

classifier_opt = GridSearchCV(classifier, grid, scoring = 'accuracy', cv=10)
classifier_opt.fit(X_train,y_train)
print("Tuned_parameter k : {}".format(classifier_opt.best_params_))
print("Best Score: {}".format(classifier_opt.best_score_))



# Random forest
random_forest = RandomForestClassifier(n_estimators=500, max_depth= 5,
                                       random_state=0).fit(X_train,y_train)
print("Test Accuracy : {:.2f} %".format(accuracy_score(random_forest.predict(X_test),y_test)))

preds = random_forest.predict(X_test)
cm = confusion_matrix(y_test, preds)
score = roc_auc_score(y_test, preds)
print("AUC :",score)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy : ",accuracies.mean())
print(cm)


from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=0.01,n_estimators=500,use_label_encoder=False,random_state=420).fit(X_train,y_train)

print("Train Accuracy : {:.2f} %".format(accuracy_score(model.predict(X_train),y_train)))

preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds)
score = roc_auc_score(y_test, preds)
print(score)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(cm)


#Performance metrics for classification
def performance(start, pred, ytest, m, xtest, s):
    print('------------------------------------',m,'------------------------------------')
    print('Accuracy',np.round(accuracy_score(pred,ytest),4))
    print('----------------------------------------------------------')
    print('Mean of Cross Validation Score',np.round(s.mean(),4))
    print('----------------------------------------------------------')
    print('AUC_ROC Score',np.round(roc_auc_score(ytest,m.predict_proba(xtest)[:,1]),4))
    print('----------------------------------------------------------')
    print('Confusion Matrix')
    print(confusion_matrix(pred,ytest))
    print('----------------------------------------------------------')
    print('Classification Report')
    print(classification_report(pred,ytest))
    print('----------------------------------------------------------')
    print('Model runtime')
    print(datetime.now()-start)
    

#List of models to try
models=[LogisticRegression(),DecisionTreeClassifier(), LGBMClassifier(learning_rate = 0.05),
        RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),XGBClassifier(verbosity=0)]


#Creates and trains model from the models list
def createmodel(trainx,testx,trainy,testy):
    for i in models:
        start=datetime.now()
        model=i
        model.fit(trainx,trainy)
        pred=model.predict(testx)
        score=cross_val_score(estimator = model, X = trainx, y = trainy, cv = 10)
        performance(start,pred,testy,model,testx,score) 



createmodel(X_train,X_test,y_train,y_test)


#It depends on dimenionality, relationship, explainablity and the requirement

#HPT for XGB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

params={
 "learning_rate"    : [0.01,0.05, 0.10, 0.15, ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,0.5],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}

m = RandomizedSearchCV(XGBClassifier(),params,cv=10)
m.fit(X_train,y_train)

print(m.best_params_)
print(m.best_estimator_)
print(m.best_score_)



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#HPT for LGBM
params={
 "learning_rate"    : [0.01,0.05, 0.10, 0.15, ] ,
 "n_estimators"     : [100,200,300,400,500],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "num_leaves"       : [15,20,25,30,35,40],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}

m = RandomizedSearchCV(LGBMRegressor(),params,cv=10)
m.fit(X_train,y_train)

print(m.best_params_)
print(m.best_estimator_)
print(m.best_score_)



#HPT for RF
params={'n_estimators':[100, 200, 300, 400, 500],
            'criterion':['gini','entropy'],
            'max_depth':[None,1,2,3,4,5,6,7,8,9,10],
           'max_features':['int','float','auto','log2']}

m = RandomizedSearchCV(RandomForestClassifier(),params,cv=10)
m.fit(X_train,y_train)

print(m.best_params_)
print(m.best_estimator_)
print(m.best_score_)



# Xgboost seems to be performing better than all other models
import numpy as np
m = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=15,
              min_child_weight=1, missing=np.nan, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

start = datetime.now()
m.fit(X_train,y_train)
pred=m.predict(X_test)
score=cross_val_score(m,X,y,cv=10)

performance(start,pred,y_test,m,X_test,score)


pd.DataFrame({"actual" :list(y_test), "predicted": list(pred)}).to_csv("test1s.csv")



print(classification_report(y_test,pred))
matrix = confusion_matrix(y_test, pred)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

importances = m.feature_importances_
feature_imp = np.array(importances)
feature_names= np.array(X.columns)
data={'feature_names':feature_names,'feature_importance':feature_imp}
table = pd.DataFrame(data) 
table.sort_values(by=['feature_importance'], ascending=False,inplace=True) 
plt.figure(figsize=(8,10))
sns.barplot(x=table['feature_importance'][:15], y=table['feature_names'][:15])
plt.title(' VARIABLE IMPORTANCE')
plt.xlabel('Feature importance')
plt.ylabel('Features')


import shap
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)


fpr, tpr, threshold = roc_curve(y_test, pred)
auc_score = roc_auc_score(y_test, pred)
fig, ax = plt.subplots(figsize = (8,8))
plt.plot(fpr, tpr, color='red',
          label='Xgboost(AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc="lower right")
plt.show()



