import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def SVM_classifier_with_tuning(X, y):
    param_grid = {
    'C': [0.1, 1, 10],  
    'gamma': ['scale', 0.01, 0.1],  
    'kernel': ['rbf']
    }

    # Suddivisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Standardizzazione delle feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Addestramento del modello SVM con GridSearchCV
    svm_model = SVC()
    grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

    # Esegui la ricerca dei migliori parametri
    grid_search.fit(X_train_scaled, y_train)

    # Stampiamo i migliori parametri trovati
    print("Migliori parametri trovati:", grid_search.best_params_)
    print("Migliore accuratezza (cross-validation):", grid_search.best_score_)

    # Usa il miglior modello trovato
    best_svm = grid_search.best_estimator_
    
    # Predizioni su training e test set
    # y_train_pred = best_svm.predict(X_train_scaled)
    y_test_pred = best_svm.predict(X_test_scaled)

    # Valutazione del modello
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\nSVM Model Performance:")
    #print(f'Accuratezza sul TRAIN: {train_accuracy:.3f}')
    print(f'Accuratezza sul TEST: {test_accuracy:.3f}')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

def SVM_classifier(X, y, C=1, gamma='scale', kernel='rbf'):


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
    svm_model.fit(X_train, y_train)

    # Predizioni
    # y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)

    # Valutazione del modello
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # return test_accuracy
    print("\nSVM Model Performance:")
    # print(f'Accuratezza sul TRAIN: {train_accuracy:.3f}')
    print(f'Accuratezza sul TEST: {test_accuracy:.3f}')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
