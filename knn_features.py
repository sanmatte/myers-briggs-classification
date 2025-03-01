
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def better_threshold(X, y, classifier, **params):
    """
    Trova la miglior feature selection per un classificatore

    Args:
        X (pd.DataFrame): Dataset di attributi.
        y (pd.Series): Serie di classi.
        classifier (class): Classe del modello di apprendimento automatico.
        **params: Parametri iperparametrici del classificatore.

    Returns:
        threshold (float): Soglia di varianza.
    """
    # Calcola la varianza di ogni colonna
    variances = X.var()
    print(variances)
    print(type(variances))
    mediana = variances.median()
    print(f'Mediana delle varianze: {mediana}')

    # Valori di soglia più significativi
    thresholds = [0.134, 0.1364, 0.1365 ,2.000, 2.100, 2.199]
    best_accuracy = 0
    best_threshold = 0
    for threshold in thresholds:
        # Seleziona le colonne con varianza maggiore della soglia
        selected_columns = variances[variances > threshold].index.tolist()
        X_selected = X[selected_columns]
        print(f'\n--- Soglia {threshold} ---')
        print(f'N° Colonne selezionate: {len(selected_columns)}')

        # Inizializza il classificatore con i parametri forniti
        train_accuracy , test_accuracy = classifier(X_selected, y, **params)

        current_accuracy = test_accuracy

        print(f'Accuratezza sul TRAIN: {train_accuracy: }')
        print(f'Accuratezza sul TEST: {current_accuracy: }')

        best_accuracy = max(best_accuracy, current_accuracy)
        best_threshold = threshold if best_accuracy == current_accuracy else best_threshold

    selected_columns = variances[variances > best_threshold].index.tolist()
    X_selected = X[selected_columns]
    return best_threshold, X_selected
    



# def kNN_classifier_with_Feature_Selection(X, y):
        
#     # Calcola la varianza di ogni colonna
#     variances = X.var()
#     mediana = variances.median()
#     print(f'Mediana delle varianze: {mediana}')

#     # ----------------   KNN with ALL features  ----------------
#     # Dividi il dataset in train e test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#     # Addestra il modello KNN
#     knn = KNeighborsClassifier(n_neighbors=4)
#     knn.fit(X_train, y_train)  # Usa tutte le caratteristiche

#     # Effettua predizioni
#     y_train_pred = knn.predict(X_train)
#     y_test_pred = knn.predict(X_test)

#     # Calcola l'accuratezza
#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     test_accuracy = accuracy_score(y_test, y_test_pred)

#     print("KNN with ALL features")
#     print(f'Accuratezza sul TRAIN: {train_accuracy: }')
#     print(f'Accuratezza sul TEST: {test_accuracy: }\n')

#     # Valori di soglia più significativi
#     thresholds = [2.000, 2.100, 2.159, 2.199]

#     for threshold in thresholds:
#         # ----------------   KNN with FEATURE SELECTION  ----------------

#         # Seleziona le colonne con varianza maggiore di 0.5
#         selected_columns = variances[variances > threshold].index.tolist()
#         X_selected = X[selected_columns]
#         print(f'N° Colonne selezionate: {len(selected_columns)}')

#         # Dividi il dataset in training e test set (con le colonne selezionate)
#         X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.20, random_state=42)

#         # Addestra il modello KNN con le feature selezionate
#         knn = KNeighborsClassifier(n_neighbors=4)
#         knn.fit(X_train, y_train)

#         # Effettua predizioni
#         y_train_pred = knn.predict(X_train)
#         y_test_pred = knn.predict(X_test)

#         # Calcola l'accuratezza
#         feature_train_accuracy = classification_report(y_train, y_train_pred)
#         feature_test_accuracy = classification_report(y_test, y_test_pred)

#        # Stampa i report
#         print(f"KNN Report:")
#         print(f'Metriche sul TRAIN:\n{feature_train_accuracy}')
#         print(f'Metriche sul TEST:\n{feature_test_accuracy}\n')
    


