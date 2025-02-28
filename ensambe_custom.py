import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def majority_voting(label, proba, voting, w=None):
    if voting == "hard":    #Hard voting
        votes = label[np.argmax(proba, axis=0)]# estrae i voti 
        if w is None: # peso uniforme
            unique, counts = np.unique(votes, return_counts=True) #conta i voti
            index = counts.argmax() # estrae il voto massimo
        else: # voti pesati
            w_votes = []
            for i, v in enumerate(votes):
                w_votes = w_votes + [v]* w[i]
            unique, counts = np.unique(w_votes, return_counts=True)
            index = counts.argmax()
        return unique[index]
    else: #soft voting
        if w is None: # peso uniforme
            s_prob = np.sum(proba, axis=0)
            index = s_prob.argmax()
        else: # voti pesati
            w_s_prob = np.sum(proba*w, axis=0)
            index = w_s_prob.argmax()
        return label[index]

class Ensemble_custom:
    def __init__(self, classifiers, voting="hard", w=None):
        self.classifiers = classifiers
        self.voting = voting
        self.w = w
        self.fitted = False

    def fit(self, x, y, labels):
        self.labels = labels
        for estimator in self.classifiers:
            sub_train_x, _, sub_train_y, _ =  train_test_split(x, y, test_size=0.20, stratify=y)
            estimator.fit(sub_train_x, sub_train_y)
        self.fitted = True

    def predict(self, test_x):
        if self.fitted:
            proba = []
            for estimator in self.classifiers:
                proba.append(estimator.predict_proba(test_x))  # Ottiene la probabilità di ogni classificatore
            proba = np.array(proba)  # Converte in array numpy con forma (n_classifiers, n_samples, n_classes)
            
            pred_y = []
            for i in range(len(test_x)):  
                pred_y.append(majority_voting(self.labels, proba[:, i, :].T, self.voting, self.w))  # NOTA: corretto uso di proba[:, i, :]
            
            return np.array(pred_y)  # Ritorna un array numpy per compatibilità con sklearn
        else:
            print("Il classificatore non è ancora stato addestrato")

#Esempio di utilizzo
def ensamble_classifiers(X, y):
    labels = np.unique(y)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, stratify=y)
    print('Shape training set:', train_x.shape)
    print('Shape validation set:', test_x.shape)
    kNN_clf = KNeighborsClassifier(1)
    dTree_clf = DecisionTreeClassifier(random_state=0)
    gNB_clf = GaussianNB()
    #notare che in questo caso i classificatori sono forniti come lista semplice
    e2_clf = Ensemble_custom(classifiers=[ kNN_clf, dTree_clf, gNB_clf], voting='soft', w=[5, 1, 5])
    e2_clf.fit(train_x, train_y, labels)
    pred_test_y = e2_clf.predict(test_x)
    print("Accuracy: ", accuracy_score(test_y, pred_test_y))
