import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plot

def random_forest_classifier(X, y):
    # Dividi il dataset in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    plot.distribution_chart(y_train)
    # Definisci il modello Random Forest con iperparametri fissi (scelti tramite tuning)
    rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=16, min_samples_split=2, min_samples_leaf=1)
    rf.fit(X_train, y_train)

    # Effettua le predizioni
    y_pred = rf.predict(X_test)

    # Calcola l'accuratezza
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")


def random_forest_classifier_with_tuning(X, y):
    # Dividi il dataset in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Definisci i parametri da testare come iperparametri
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': list(map(int, list(range(16, 20)))), # [16, 17, 18, 19]
        'min_samples_split': list(map(int, list(range(1, 6)))), # [1, 2, 3, 4, 5]
        'min_samples_leaf': list(map(int, list(range(1, 5)))) # [1, 2, 3, 4]
    }

    # Definisci il modello Random Forest
    rf = RandomForestClassifier(random_state=42)

    # Cerca i migliori iperparametri e addestra il modello con essi
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    # Ottieni il miglior modello
    best_rf = grid_search.best_estimator_

    # Effettua le predizioni
    y_pred = best_rf.predict(X_test)

    # Calcola l'accuratezza
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Random Forest Accuracy: {accuracy:.4f}")
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)


