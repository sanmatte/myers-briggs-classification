from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

#load dataset
import pandas as pd
df = pd.read_csv('16P.csv')
def compute_accuracy(pred_y, test_y):
    return (pred_y == test_y).sum() / len(pred_y)


#sample dataset
df = df.sample(frac=0.5, random_state=42)
# Separate features and target
X = df.drop(columns=['Personality', 'Response Id'], axis=1)  # Adjust if column name is different
y = df['Personality']

#split con stratificazione
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)
import utils.plot as plot
plot.distribution_chart(y)

# frequenza delle classi
print('Frequenza delle classi nel dataset originale %s' % sorted(Counter(y).items()))


dTree_clf = DecisionTreeClassifier(random_state=0)
dTree_clf.fit(train_x, train_y)
pred_y = dTree_clf.predict(test_x)
print('Accuratezza del dtree sul dataset originale %s' % compute_accuracy(test_y, pred_y))

rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(train_x, train_y)
print('Dataset dopo il random undersampling %s' % sorted(Counter(y_resampled).items()))  

dTree_clf = DecisionTreeClassifier(random_state=0)
dTree_clf.fit(X_resampled, y_resampled)
pred_y = dTree_clf.predict(test_x)
print('Accuratezza del dtree dopo random undersampling %s' % compute_accuracy(test_y, pred_y))
# from sklearn.preprocessing import LabelEncoder
# # Converti le classi in numeri
# le = LabelEncoder()
# y_numeric = le.fit_transform(y)

# # Ora usa y_numeric invece di y
# iht = InstanceHardnessThreshold()
# X_resampled, y_resampled = iht.fit_resample(X, y_numeric)

# # Se vuoi riconvertire i numeri in etichette originali dopo il resampling:
# y_resampled = le.inverse_transform(y_resampled)
# print('Dataset dopo il probabilistico undersampling %s' % sorted(Counter(y_resampled).items()))  

# dTree_clf = DecisionTreeClassifier(random_state=0)
# dTree_clf.fit(X_resampled, y_resampled)
# pred_y = dTree_clf.predict(test_x)
# print('Accuratezza del dtree dopo probabilistico undersampling %s' % compute_accuracy(test_y, pred_y))
#y_resampled = le.inverse_transform(y_resampled)
#X_resampled, y_resampled = rus.fit_resample(train_x, train_y)
# nm = NearMiss(version=1)
# X_resampled, y_resampled = nm.fit_resample(X, y)
# print('Dataset dopo il NearMiss_v1 undersampling %s' % sorted(Counter(y_resampled).items()))

# dTree_clf = DecisionTreeClassifier(random_state=0)
# dTree_clf.fit(X_resampled, y_resampled)
# pred_y = dTree_clf.predict(test_x)
# print('Accuratezza del dtree dopo NearMiss_v1 undersampling %s' % compute_accuracy(test_y, pred_y))

# nm = NearMiss(version=2)
# X_resampled, y_resampled = nm.fit_resample(X, y)
# print('Dataset dopo il NearMiss_v2 undersampling %s' % sorted(Counter(y_resampled).items()))

# dTree_clf = DecisionTreeClassifier(random_state=0)
# dTree_clf.fit(X_resampled, y_resampled)
# pred_y = dTree_clf.predict(test_x)
# print('Accuratezza del dtree dopo NearMiss_v2 undersampling %s' % compute_accuracy(test_y, pred_y))

cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)
print('Dataset dopo il kMeans undersampling %s' % sorted(Counter(y_resampled).items()))

dTree_clf = DecisionTreeClassifier(random_state=0)
dTree_clf.fit(X_resampled, y_resampled)
pred_y = dTree_clf.predict(test_x)
print('Accuratezza del dtree dopo kMeans undersampling %s' % compute_accuracy(test_y, pred_y))

rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(train_x, train_y)
print('Dataset dopo il random oversampling %s' % sorted(Counter(y_resampled).items()))  

dTree_clf = DecisionTreeClassifier(random_state=0)
dTree_clf.fit(X_resampled, y_resampled)
pred_y = dTree_clf.predict(test_x)
print('Accuratezza del dtree dopo random oversampling %s' % compute_accuracy(test_y, pred_y))

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print('Dataset dopo SMOTE oversampling %s' % sorted(Counter(y_resampled).items()))

dTree_clf = DecisionTreeClassifier(random_state=0)
dTree_clf.fit(X_resampled, y_resampled)
pred_y = dTree_clf.predict(test_x)
print('Accuratezza del dtree dopo SMOTE oversampling %s' % compute_accuracy(test_y, pred_y))

# ada = ADASYN(random_state=42)
# X_resampled, y_resampled = ada.fit_resample(X, y)
# print('Dataset dopo ADASYN oversampling %s' % sorted(Counter(y_resampled).items()))

# dTree_clf = DecisionTreeClassifier(random_state=0)
# dTree_clf.fit(X_resampled, y_resampled)
# pred_y = dTree_clf.predict(test_x)
# print('Accuratezza del dtree dopo ADASYN oversampling %s' % compute_accuracy(test_y, pred_y))