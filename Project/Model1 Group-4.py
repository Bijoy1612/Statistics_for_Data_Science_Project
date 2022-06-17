#CSE303 Project Model 1 : Logistic Regression
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve , roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
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
plt.figure(figsize = (20,25), dpi= 300)
plt.subplot(5,4,1)
sns.histplot(train_file['cont0'], color = 'green')
plt.subplot(5,4,2)
sns.histplot(train_file['cont1'], color = 'orange')
plt.subplot(5,4,3)
sns.histplot(train_file['cont2'], color = 'green')
plt.subplot(5,4,4)
sns.histplot(train_file['cont3'], color = 'orange')
plt.subplot(5,4,5)
sns.histplot(train_file['cont4'], color = 'orange')
plt.subplot(5,4,6)
sns.histplot(train_file['cont5'], color = 'green')
plt.subplot(5,4,7)
sns.histplot(train_file['cont6'], color = 'orange')
plt.subplot(5,4,8)
sns.histplot(train_file['cont7'], color = 'green')
plt.subplot(5,4,9)
sns.histplot(train_file['cont8'], color = 'green')
plt.subplot(5,4,10)
sns.histplot(train_file['cont9'], color = 'orange')
plt.subplot(5,4,11)
sns.histplot(train_file['cont10'], color = 'green')

plt.suptitle('Histogram of cont features', fontsize = 40, y = 0.91)
plt.savefig('Histogram of cont features.png', dpi=300, bbox_inches='tight')

#%%
plt.figure(figsize = (20,25), dpi= 300)
plt.subplot(5,4,1)
sns.histplot(train_file['cat0'], color = 'green')
plt.subplot(5,4,2)
sns.histplot(train_file['cat1'], color = 'orange')
plt.subplot(5,4,3)
sns.histplot(train_file['cat2'], color = 'green')
plt.subplot(5,4,4)
sns.histplot(train_file['cat3'], color = 'orange')
plt.subplot(5,4,5)
sns.histplot(train_file['cat4'], color = 'orange')
plt.subplot(5,4,6)
sns.histplot(train_file['cat5'], color = 'green')
plt.subplot(5,4,7)
sns.histplot(train_file['cat6'], color = 'orange')
plt.subplot(5,4,8)
sns.histplot(train_file['cat7'], color = 'green')
plt.subplot(5,4,9)
sns.histplot(train_file['cat8'], color = 'green')
plt.subplot(5,4,10)
sns.histplot(train_file['cat9'], color = 'orange')
plt.subplot(5,4,11)
sns.histplot(train_file['cat10'], color = 'green')
plt.subplot(5,4,12)
sns.histplot(train_file['cat11'], color = 'orange')
plt.subplot(5,4,13)
sns.histplot(train_file['cat12'], color = 'orange')
plt.subplot(5,4,14)
sns.histplot(train_file['cat13'], color = 'green')
plt.subplot(5,4,15)
sns.histplot(train_file['cat14'], color = 'orange')
plt.subplot(5,4,16)
sns.histplot(train_file['cat15'], color = 'green')
plt.subplot(5,4,17)
sns.histplot(train_file['cat16'], color = 'green')
plt.subplot(5,4,18)
sns.histplot(train_file['cat17'], color = 'orange')
plt.subplot(5,4,19)
sns.histplot(train_file['cat18'], color = 'green')

plt.suptitle('Histogram of cat features', fontsize = 40, y = 0.91)
plt.savefig('Histogram of Cat Features', dpi=300, bbox_inches='tight')

#%%
plt.figure(figsize=(16, 6), dpi= 500)
heatmap = sns.heatmap(train_file.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig('Corr heatmap.png', dpi=300, bbox_inches='tight')

#%%
plt.figure(figsize=(8, 12), dpi= 500)
heatmap = sns.heatmap(train_file[0:11].corr()[['target']].sort_values(by='target', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Target', fontdict={'fontsize':18}, pad=16);
plt.savefig('Corr heatmap with target.png', dpi=300, bbox_inches='tight')

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
print('Logistic Regression Model\n')
model_logistic = LogisticRegression (solver='liblinear', C =1, penalty='l1', class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None,
                   random_state=0, tol=0.0001, verbose=0,
                   warm_start=False)
model_logistic.fit(X_train,y_train)
y_pred_logistic = model_logistic.predict(X_test)

#%%
#Scores

logit_score = accuracy_score(y_test, y_pred_logistic) * 100
print("Accuracy for Logistic Regression: ", round(logit_score, 1), "%" )

p_logit = precision_score(y_test, y_pred_logistic, average='micro') * 100
print('Precision for Logistic Regression: ',round(p_logit, 1), "%")

r_logit = recall_score(y_test, y_pred_logistic, average='weighted') * 100
print('Recall for Logistic Regression:  ', round(r_logit, 1), "%")

f1_logit = f1_score(y_test, y_pred_logistic) * 100
print('F1-score for Logistic Regression: ', round(f1_logit, 1), "%")

rus_logit = roc_auc_score(y_test, y_pred_logistic) * 100
print('ROC AUC Score (Logistic Model):',round(rus_logit, 1), "%")

#%%

logit_roc_auc = roc_auc_score(y_test, model_logistic.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_logistic.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#%%
cm_logit = confusion_matrix(y_test, y_pred_logistic)
print(cm_logit)
plot_confusion_matrix(model_logistic, X_test, y_test)  
plt.title('Confusion Matrix for Logistic Model')
plt.savefig('Log_CF')
plt.show() 
#%%
#New Dataset logistic
logit =  model_logistic.predict_proba(test_file)
result = pd.read_csv('sample_submission.csv',index_col = 0)
result['target'] = logit[:,1]
result.to_csv('Group4_model1_result.csv')
print('Sucessful! Saved New File')