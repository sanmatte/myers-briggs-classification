import pandas as pd
import plot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

def KNN_classifier_with_tuning(X, y):
    # strutture dati per memorizzare le accuratezze finali 
    total_train_accuracy = []
    total_test_accuracy = []

    # numero di volte che il dataset viene diviso in training e test set (cross-validation)
    number_folds = 5
    
    # funzione che divide il dataset in training e test set, con reintroduzione e shuffle
    kf = KFold(n_splits=number_folds, shuffle=True, random_state=42)

    # range di valori per l'iperparametro n_neighbors
    neighbor_number = range(1, 6)
    
    for n_neighbors in neighbor_number:
        print(f"Neighbors: {n_neighbors}")

        # liste che memorizzano le accuratezze dei diversi modelli, addestrati, al variare dell'iperparametro n_neighbors
        training_accuracy = []
        test_accuracy = []
        
        # indice del fold
        fold = 0

        # cicla su tutti i fold
        for train_index, test_index in kf.split(X, y):
            print(f"Fold: {fold}")

            #divide il dataset in training e test set
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            # crea e addestra il classificatore KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)  # build the model
            clf.fit(X_train, y_train)

            # calcola l'accuratezza in fase di training e test 
            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            
            print([train_acc, test_acc])

            # aggiunge i valori appena calcolati alle liste, utili a stampare il plot per confronrare le curve delle due accuracy
            training_accuracy.append(train_acc) # record training set accuracy
            test_accuracy.append(test_acc)   # record generalization accuracy

            fold += 1

        # Calcola la media delle accuratezze per ogni valore di k
        total_train_accuracy.append(np.mean(training_accuracy))
        total_test_accuracy.append(np.mean(test_accuracy))

    # stampa il plot
    plot.trainingAccuracy_vs_testAccuracy_chart(neighbor_number, neighbor_number, total_train_accuracy, total_test_accuracy, "Training Accuracy", "Test Accuracy", "Number of Neighbors", "Accuracy", "Tuning KNN")
    