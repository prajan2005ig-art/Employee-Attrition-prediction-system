import pandas as pd
df = pd.read_csv('employee pred.csv')
pd.options.display.max_columns = 500
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df = df.join(pd.get_dummies(df['BusinessTravel'])).drop('BusinessTravel', axis=1)
df = df.join(pd.get_dummies(df['Department'],prefix = 'Department')).drop('Department', axis=1)
df = df.join(pd.get_dummies(df['EducationField'],prefix = 'Education')).drop('EducationField', axis=1)
df = df.join(pd.get_dummies(df['JobRole'],prefix = 'Role')).drop('JobRole', axis=1)
df = df.join(pd.get_dummies(df['MaritalStatus'],prefix = 'Mstatus')).drop('MaritalStatus', axis=1)
df = df.map(lambda x: 1 if x is True else 0 if x is False else x)
df = df.drop('EmployeeNumber', axis=1)
#import matplotlib.pyplot as plt
#df.hist(figsize = (20, 15))
#plt.tight_layout()
#plt.show()
df = df.drop(['EmployeeCount','Over18','StandardHours'], axis=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
X,y = df.drop('Attrition', axis=1), df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42, stratify = y)
model = RandomForestClassifier(n_estimators = 100, random_state = 42,class_weight = "balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.fit(X_train, y_train)
print("Model training completed")
y_pred = model.predict(X_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


