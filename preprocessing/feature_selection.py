
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