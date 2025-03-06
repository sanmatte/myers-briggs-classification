from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def random_forest_classifier(X, y, n_estimators=200, max_depth=16, min_samples_split=2, min_samples_leaf=1):
    # Divide il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Definire e Trainare il modello con iperparametri fissi (scelti tramite tuning)
    rf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    rf.fit(X_train, y_train)

    # Predizioni
    #y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Valutazione del modello - Calcola 'Accuracy on training set' e 'Accuracy on test set'
    train_accuracy = 'Not Calculated'
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # return 'Training Accuracy' e 'Test Accuracy'
    return test_accuracy, train_accuracy


def random_forest_classifier_with_tuning(X, y):
    # Divide il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Definisci i parametri da testare come iperparametri
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': list(range(16, 20)), # [16, 17, 18, 19]
        'min_samples_split': list(range(1, 6)), # [1, 2, 3, 4, 5]
        'min_samples_leaf': list(range(1, 5)) # [1, 2, 3, 4]
    }

    # Crea un nuovo modello RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Cerca i migliori iperparametri e addestra il modello con essi
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    # Ottiene il miglior modello
    best_rf = grid_search.best_estimator_
    
    # Predizioni
    #y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)

    # Valutazione del modello - Calcola 'Accuracy on training set' e 'Accuracy on test set'
    train_accuracy = 'Not Calculated'
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Best Random Forest Accuracy: {test_accuracy}")
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)