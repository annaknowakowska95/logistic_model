# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:51:28 2021

@author: anowakowska
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('BankChurners.csv')

del df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1']
del df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']



num_labels={'Attrition_Flag': {"Existing Customer": 1, "Attrited Customer": 0},
            "Gender": {"M": 1, "F": 0},
            'Education_Level': {"Unknown": 0, "Uneducated": 1, "High School": 2, "Graduate": 3, "Post-Graduate": 4, "College": 5, "Doctorate": 6},
            'Marital_Status': {"Unknown": 0, "Single": 1, "Married": 2, "Divorced": 3},
            'Income_Category': {"Unknown": 0, "Less than $40K": 1, "$40K - $60K": 2, "$60K - $80K": 3, "$80K - $120K": 4, "$120K +": 5},
            'Card_Category': {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3}}


df=df.replace(num_labels)
 


X=df.iloc[:,2:20].values
y=df['Attrition_Flag']

c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

logreg = LogisticRegression(max_iter=10000)
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

logreg_cv.fit(X_train, y_train)

y_pred = logreg_cv.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
    
y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

