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

    print("STARTING BEST FEATURE SELECTION THRESHOLD CALCULATOR\n")

    variances = X.var()
    mediana = variances.median()
    print(f'Mediana delle varianze: {mediana}')

    # Valori di soglia più significativi
    thresholds = [0.134, 0.1364,2.000, 2.100, 2.199]
    best_accuracy = 0
    best_threshold = 0
    for threshold in thresholds:
        # Seleziona le colonne con varianza maggiore della soglia
        selected_columns = variances[variances > threshold].index.tolist()
        X_selected = X[selected_columns]
        print(f'\n--- Threshold {threshold} ---')
        print(f'N° Colonne selezionate: {len(selected_columns)}')

        #train_accuracy = "[Not Calculated to reduce complexity time]"
        # Chiama il classificatore con le colonne selezionate
        test_accuracy, train_accuracy = classifier(X_selected, y, **params)

        current_accuracy = test_accuracy

        #print(f'Accuracy on training set: {train_accuracy}')
        print(f'Accuracy on test set: {current_accuracy}')

        best_accuracy = max(best_accuracy, current_accuracy)
        best_threshold = threshold if best_accuracy == current_accuracy else best_threshold

    selected_columns = variances[variances > best_threshold].index.tolist()
    X_selected = X[selected_columns]
    return best_threshold, X_selected