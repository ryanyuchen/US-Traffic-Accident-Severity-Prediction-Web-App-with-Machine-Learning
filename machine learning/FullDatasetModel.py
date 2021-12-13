# Import numpy, pandas, matpltlib.pyplot, sklearn modules and seaborn
import numpy as np
import pandas as pd
import io
import re
import os
import time

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

#Import Others
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

#Load Data
df = pd.read_csv('CleanData_v2.csv')

#Drop unname column
df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
#rows = df.sample(frac=0.01,random_state=1) #======================YL
df.info()

df["Severity"].value_counts()

#Resampling
df_bl = pd.concat([df[df['Severity']==4].sample(1500000, random_state=42,replace=True),
                   df[df['Severity']==2].sample(750000, random_state=42,replace=True),
                   df[df['Severity']==3].sample(75000, random_state=42,replace=True),
                   df[df['Severity']==1].sample(22500, replace = True, random_state=42)], axis=0)

df_bl["Severity"].value_counts()

# Generate dummies for categorical data
cat = ['Side','State','Timezone','Wind_Direction', 'Weekday', 'Month', 'Hour','Sunrise_Sunset']
df_bl[cat] = df_bl[cat].astype('category')
df_bl = pd.get_dummies(df_bl, columns=cat, drop_first=True)
df_bl.head()

#df = df_dummy
X = df_bl.drop('Severity',axis=1)
y = df_bl['Severity']


# split train test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(\
              X, y, test_size=0.20, random_state=42, stratify=y)


# --------------Random Forest algorithm--------------------------
tic1 = time.time()
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=150)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Get the f1 score
f1_clf=f1_score(y_test, y_pred, average='weighted')

# Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))
print("[Randon forest algorithm] f1_score: {:.3f}.".format(f1_clf))
toc1 = time.time()
print('Elapsed time for Randon forest is %f seconds \n' % float(toc1 - tic1))

   
#Algorithm Random Forest
#Visualize important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
pd.set_option('display.max_rows', 350)
pd.set_option('display.max_columns', 350)
plt.style.use('ggplot')

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k=20
sns.barplot(x=feature_imp[:20], y=feature_imp.index[:k])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# List top k important features
k=20
feature_imp.sort_values(ascending=False)[:k]

#Algorithm Random Forest
#Select the top important features, set the threshold
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.03
sfm = SelectFromModel(clf, threshold=0.03)

# Train the selector
sfm.fit(X_train, y_train)

feat_labels=X.columns

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature Model
print('[Randon forest algorithm -- Full feature] accuracy_score: {:.3f}.'.format(accuracy_score(y_test, y_pred)))

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature Model
print('[Randon forest algorithm -- Limited feature] accuracy_score: {:.3f}.'.format(accuracy_score(y_test, y_important_pred)))

#View The F1 Score of Our Full Feature Model
print('[Randon forest algorithm -- Full feature] f1_score: {:.3f}.'.format(f1_score(y_test, y_pred, average='weighted')))

# View The F1 Score Of Our Limited Feature Model
print('[Randon forest algorithm -- Limited feature] f1_score: {:.3f}.'.format(f1_score(y_test, y_important_pred, average='weighted')))


#--------------- XGBoost algorithm
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
tic7 = time.time()
xgb = XGBClassifier(objective= 'multi:softmax',num_class=4,n_fold=4,
                    colsample_bytree = 1,
                    learning_rate = 0.15,
                    n_estimators = 600,
                    subsample = 0.3)
xgb.fit(X_train, y_train)

y_pred=xgb.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Get f1 score
f1_xgb=f1_score(y_test, y_pred, average='weighted')


print("[XGBoost algorithm] accuracy_score: {:.3f}.".format(acc))
print("[XGBoost algorithm] f1_score: {:.3f}.".format(f1_xgb))
toc7 = time.time()
print('Elapsed time for XGBoost is %f seconds \n' % float(toc7 - tic7))

#==============YL
pd.set_option('display.max_rows', 350)
pd.set_option('display.max_columns', 350)
plt.style.use('ggplot')

feature_imp = pd.Series(xgb.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k=20
sns.barplot(x=feature_imp[:20], y=feature_imp.index[:k])
# Add labels to your graph
plt.xlabel('Feature Importance Score-XGB')
plt.ylabel('Features-XGB')
plt.title("Visualizing Important Features-XGB")
plt.legend()
plt.show()