# Data Pre-processing

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier


df = pd.read_csv("./data/Crash_Details_Table.csv")
print(df.head(5))
print(df.columns)
df.info()
# EDA on the data
print(df['FATAL'].value_counts())
# So we know we are dealing with very imbalanced data here.
# We will probably have to do techniques like SMOTE to try to balance the data

# Let's check some other values
print(df['AGE'].value_counts())
print(df['MAJORINJURY'].value_counts())
print(df['MINORINJURY'].value_counts())
only_deaths = df[df['FATAL'] == 'Y']
print(only_deaths['MAJORINJURY'].value_counts())
print(only_deaths['MINORINJURY'].value_counts())
# Ok, interesting, it looks like MAJORINJURY and MINORINJURY are all N for FATAL... they are probably correlated then
# and calculated off one another so I should remove it.

# Create a new variable for "fatality within the same CRIMEID"
#y_n = {"N":0, "Y":1}
#df['FATAL'] = df['FATAL'].map(y_n)
#df.sort_values(by=["CRIMEID", "FATAL"], ascending=False)
#df["FATALINEVENT"] = df[df.groupby(by=["CRIMEID"]).sum() > 0]
# Let's do some recoding of the variables here
# Label Encode the target variable
class_le = LabelEncoder()
y = class_le.fit_transform(df['FATAL'])
print("Target variable after label encoding: ")
print(y)

y_inv = class_le.inverse_transform(y)

print("Class variable inverse transform: ")
print(y_inv)
# Make Age Castegorical
# y_n = {"N":0, "Y":1}
# df['AGE'] = df['AGE'].map(y_n)
# df["AGE_CAT"]
# Make independent variables into categorical variables
df_copy = df.copy(deep=True)
df_copy = df_copy.drop(columns=["OBJECTID", "CRIMEID", "CCN", "PERSONID", "VEHICLEID", "AGE"])
df_copy[['PERSONTYPE', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']] = df_copy[['PERSONTYPE', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']].apply(class_le.fit_transform)
df_copy[['PERSONTYPE', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']] = df_copy[['PERSONTYPE', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']].astype('object')
df_X = df_copy.drop(columns='FATAL')
df_X.info()

enc = OneHotEncoder(sparse=False)
x = enc.fit_transform(df_X[['PERSONTYPE', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
unique, counts = np.unique(y_test, return_counts=True)
print('Y-Test Unique', unique, "and counts", counts)
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.show()
print('Y-Train Unique', unique, "and counts", counts)
### PERFORM SAMPLING AND SMOTE TO BALANCE ###
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
over = SMOTE(random_state=1, sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.4)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X_train, y_train = pipeline.fit_resample(X_train, y_train)
unique, counts = np.unique(y_train, return_counts=True)

plt.bar(unique, counts)
plt.show()

#np.save("x.npy", x); np.save("y.npy", y)
# Only scale continuous vars if we have them
# stdsc = StandardScaler()
#
# stdsc.fit(arr_for_model)
#
# X_std = stdsc.transform(arr_for_model)
#
# print("Values after standard scaling: ")
# print(X_std)

# clf = MLPClassifier(solver='sgd',
#                             alpha=.001,
#                             hidden_layer_sizes=(5, 5),
#                             random_state=1)
# # Train the model using the training sets
# clf.fit(X_train, y_train.flatten())
# # Predict the response for test dataset
# y_pred = clf.predict(X_test)
# # train model
# #rfc = RandomForestClassifier(n_estimators=100, class_weight="balanced").fit(X_train, y_train.flatten())
# y_pred = clf.predict(X_test)
#
# cmx_MLP = confusion_matrix(y_test, y_pred)
# print(cmx_MLP)
# cfrp = classification_report(y_test, y_pred)
# print(cfrp)
# print(metrics.f1_score(y_test, y_pred))

rfc = RandomForestClassifier(n_estimators=50).fit(X_train, y_train.flatten())
y_pred = rfc.predict(X_test)

cmx_MLP = confusion_matrix(y_test, y_pred)
print(cmx_MLP)
cfrp = classification_report(y_test, y_pred)
print(cfrp)
print(metrics.f1_score(y_test, y_pred))

importance = rfc.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

rfc = RandomForestClassifier(n_estimators=50, class_weight="balanced").fit(X_train, y_train.flatten())
y_pred = rfc.predict(X_test)

cmx_MLP = confusion_matrix(y_test, y_pred)
print(cmx_MLP)
cfrp = classification_report(y_test, y_pred)
print(cfrp)
print(metrics.f1_score(y_test, y_pred))

importance = rfc.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# model = RandomForestClassifier(n_estimators=50, class_weight='balanced')
# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# scores = cross_val_score(model, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# # summarize performance
# print('Mean ROC AUC: %.3f' % np.mean(scores))
#
# model = RandomForestClassifier(n_estimators=50, class_weight='balanced_subsample')
# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# scores = cross_val_score(model, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# # summarize performance
# print('Mean ROC AUC: %.3f' % np.mean(scores))

print('end')