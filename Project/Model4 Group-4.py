#CSE303 Project Model 3 : Ridge Regression
#%%
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve , roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import Lasso

#%%
#reading data files
train_file = pd.read_csv('train.csv')
test_file = pd.read_csv('test.csv')
#%%
print(train_file.columns)
print(test_file.columns)
#%%
print(train_file.info)
print(test_file.info)
#%%
print('Train File null Valuse:\n',train_file.isna().sum()) #No null value found
print('Test File null Valuse:\n',test_file.isna().sum()) #No null value found
train_file = train_file.dropna()
test_file = test_file.dropna()
#%%
print('Train File unique Values:\n',train_file.nunique())
print('Test File unique Values:\n',test_file.nunique())

#%%
train_file = train_file.drop_duplicates()
test_file = test_file.drop_duplicates()

#%%
encoder = LabelEncoder()
scaler = StandardScaler()
#%%
train_file.drop(columns = ['id'] , inplace = True)
test_file.drop(columns = ['id'] , inplace = True)

#%%
column = list(test_file.columns)
print(column)
#%%
for i in column:
    encoder.fit(list(train_file[i].values) + list(test_file[i].values))
    train_file[i]= encoder.transform(train_file[i])
    test_file[i] = encoder.transform(test_file[i])

#%%
X = train_file.drop(columns = ['target'])
y = train_file['target']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80, shuffle = False)

#%%
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
test_file[test_file.columns] = scaler.transform(test_file[test_file.columns])
#%%
print('Training data:\n', train_file.head())
print('Testing data:\n', test_file.head())

#%%
#Lasso Regression

model_lasso = Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
              normalize=False, positive=False, precompute=False, random_state=None,
              selection='random', tol=0.0001, warm_start=True)

model_lasso.fit(X_train, y_train)

y_predict_lasso = abs(np.around(model_lasso.predict(X_test),0)) 

#%%
lasso_acc_score = accuracy_score(y_test, y_predict_lasso) * 100
print("Accuracy for Lasso Regression: ", round(lasso_acc_score, 1), "%" )

p_lasso = precision_score(y_test, y_predict_lasso, average='micro') * 100
print('Precision for Lasso Regression: ',round(p_lasso, 1), "%")

r_lasso = recall_score(y_test, y_predict_lasso, average='weighted') * 100
print('Recall for Lasso Regression:  ', round(r_lasso, 1), "%")

f1_lasso = f1_score(y_test, y_predict_lasso) * 100
print('F1-score for Lasso Regression: ', round(f1_lasso, 1), "%")

rus_lasso = roc_auc_score(y_test, y_predict_lasso) * 100
print('ROC AUC Score (Lasso Model):',round(rus_lasso, 1), "%")

#%%
logit_roc_auc = roc_auc_score(y_test, model_lasso.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_lasso.predict(X_test))
plt.figure()
plt.plot(fpr, tpr, label='Lasso Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Lasso_ROC')
plt.show()

#%%
cm_lasso = confusion_matrix(y_test, y_predict_lasso)
print(cm_lasso)
plot_confusion_matrix(model_lasso, X_test, y_test)
plt.title('Confusion Matrix for Lasso Model')
plt.savefig('Log_Lasso')
plt.show() 
#%%
#New Dataset
lass =  model_lasso.predict(test_file)
result3 = pd.read_csv('sample_submission.csv',index_col = 0)
result3['target'] = lass
result3.to_csv('group4_model4_result.csv')
print('Sucessful! Saved New File')
