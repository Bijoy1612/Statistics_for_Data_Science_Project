#CSE303 Project Model 3 : Ridge Regression
#%%
#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve , roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier

#%%
#reading data files
train_file = pd.read_csv('train.csv')
test_file = pd.read_csv('test.csv')
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
test_file.drop(columns = [ 'id'] , inplace = True)

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
#Ridge Regression
print('Ridge Regression Model\n')
model_ridge = RidgeClassifier(alpha = 0.1,solver = 'sag')
model_ridge.fit(X_train,y_train)

y_pred_ridge = model_ridge.predict(X_test)

#%%
ridge_acc_score = accuracy_score(y_test, y_pred_ridge) * 100
print("Accuracy for Ridge Regression: ", round(ridge_acc_score, 1), "%" )

p_ridge = precision_score(y_test, y_pred_ridge, average='micro') * 100
print('Precision for Ridge Regression: ',round(p_ridge, 1), "%")

r_ridge = recall_score(y_test, y_pred_ridge, average='weighted') * 100
print('Recall for Ridge Regression:  ', round(r_ridge, 1), "%")

f1_ridge = f1_score(y_test, y_pred_ridge) * 100
print('F1-score for Ridge Regression: ', round(f1_ridge, 1), "%")

rus_ridge = roc_auc_score(y_test, y_pred_ridge) * 100
print('ROC AUC Score (Ridge Model):',round(rus_ridge, 1), "%")

#%%
ridge_roc_auc = roc_auc_score(y_test, model_ridge.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_ridge.predict(X_test))
plt.figure()
plt.plot(fpr, tpr, label='Ridge Regression (area = %0.2f)' % ridge_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Ridge_ROC')
plt.show()
#%%
cm_ridge = confusion_matrix(y_test, y_pred_ridge)
print(cm_ridge)
plot_confusion_matrix(model_ridge, X_test, y_test) 
plt.title('Confusion Matrix for Ridge Model')
plt.savefig('Log_Ridge') 
plt.show() 
#%%
#New Dataset
rid =  model_ridge.predict(test_file)
result2 = pd.read_csv('sample_submission.csv',index_col = 0)
result2['target'] = rid
result2.to_csv('group4_model3_result.csv')
print('Sucessful! Saved New File')