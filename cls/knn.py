from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import utils.plot as plot

def KNN_classifier(X, y, k=4):

    # Divide il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Definire e Trainare il modello con iperparametri fissi (scelti tramite tuning)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    # Predizioni
    #y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Valutazione del modello - Calcola 'Accuracy on training set' e 'Accuracy on test set'
    train_accuracy = 'Not Calculated'
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # return 'Training Accuracy' e 'Test Accuracy'
    return test_accuracy, train_accuracy


def KNN_classifier_with_tuning(X, y):
    # range di valori per l'iperparametro n_neighbors
    neighbor_number = range(1, 6)

    # liste che memorizzano le accuratezze dei diversi modelli, addestrati, al variare dell'iperparametro n_neighbors - utili al plot
    total_training_accuracy = []
    total_test_accuracy = []

    for n_neighbors in neighbor_number:
        print(f"Neighbors: {n_neighbors}")
        
        # Crea e addestra un kNN clf con iperparametro = n_neighbors
        # Verificare che KNN_classifier() stia restituendo correttamente la lista di training (commentato per evitare sovracarico computazionale)
        test_accuracy, training_accuracy = KNN_classifier(X, y, n_neighbors)

        # Aggiunge le accuratezze ottenute ad ogni iterazione, al variare di n_neighbor - liste utili al plot
        total_training_accuracy.append(training_accuracy)
        total_test_accuracy.append(test_accuracy)

    # Stampa il valore di k con il quale il classificatore ha ottenuto le migliori prestazioni
    print(f"L'acuratezza migliore registrata è {max(total_test_accuracy)}\n Il valore migliore di k è `{total_test_accuracy.index(max(total_test_accuracy))+1}`")

    # stampa il plot
    plot.trainingAccuracy_vs_testAccuracy_chart(neighbor_number, neighbor_number, total_training_accuracy, total_test_accuracy, "Training Accuracy", "Test Accuracy", "Number of Neighbors", "Accuracy", "Tuning KNN")
