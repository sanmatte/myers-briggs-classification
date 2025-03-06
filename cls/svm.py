from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def SVM_classifier(X, y, C=1, gamma='scale', kernel='rbf'):

    # Divide il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Definire e Trainare il modello con iperparametri fissi (scelti tramite tuning)
    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
    svm_model.fit(X_train, y_train)

    # Predizioni
    # y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)

    # Valutazione del modello - Calcola 'Accuracy on training set' e 'Accuracy on test set'
    train_accuracy = 'Not Calculated'
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # return 'Training Accuracy' e 'Test Accuracy'
    return test_accuracy, train_accuracy

def SVM_classifier_with_tuning(X, y):

    param_grid = {
    'C': [0.1, 1, 10],  
    'gamma': ['scale', 0.01, 0.1],  
    'kernel': ['rbf']
    }

    # Divide il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Standardizzazione delle features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crea un nuovo modello SVC
    svm_model = SVC()

    # Cerca i migliori iperparametri e addestra il modello con essi
    grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Ottiene il miglior modello
    best_svm = grid_search.best_estimator_
    
    # Predizioni su training e test set
    # y_train_pred = best_svm.predict(X_train_scaled)
    y_test_pred = best_svm.predict(X_test_scaled)

    # Valutazione del modello
    train_accuracy = 'Not Calculated'
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Best SVM Accuracy: {test_accuracy}")
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)
