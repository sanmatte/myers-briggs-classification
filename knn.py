import pandas as pd
import plot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import StandardScaler

def KNN_classifier(X, y, k=4):
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 42, test_size = 0.2)

    # crea e addestra il classificatore KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k)  # build the model
    clf.fit(train_x, train_y)

    # calcola l'accuratezza in fase di training e test 
    test_acc = clf.score(test_x, test_y)
    print(f"Test Accuracy: {test_acc}")
    return test_acc


def KNN_classifier_with_tuning(X, y):
    # range di valori per l'iperparametro n_neighbors
    neighbor_number = range(1, 6)
    
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # liste che memorizzano le accuratezze dei diversi modelli, addestrati, al variare dell'iperparametro n_neighbors
    training_accuracy = []
    test_accuracy = []

    for n_neighbors in neighbor_number:
        print(f"Neighbors: {n_neighbors}")
        
        train_acc, test_acc = KNN_classifier(X_standardized, y, n_neighbors)

        print([train_acc, test_acc])

        # Calcola la media delle accuratezze per ogni valore di k
        training_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    # stampa il plot
    plot.trainingAccuracy_vs_testAccuracy_chart(neighbor_number, neighbor_number, training_accuracy, test_accuracy, "Training Accuracy", "Test Accuracy", "Number of Neighbors", "Accuracy", "Tuning KNN")

