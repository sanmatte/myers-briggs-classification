import plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

def DecisionTree_with_tuning(X, y):

    # Liste che memorizzano le accuracy dei diversi modelli, addestrati, al variare dell'iperparametro max_depth
    acc_train = []
    acc_test = []

    # ----------------  DECISION TREE with ALL features  ----------------
    # Tuning dell'iperparametro max_depth
    print("DECISION TREE with ALL features")
    max_depth_range = list(range(1, 20))
    for depth in max_depth_range:
        # Fit depth-esimo DecisionTreeClassifier con max_depth = depth
        dTree_clf = DecisionTreeClassifier(criterion="gini", max_depth = depth)

        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 42, test_size=0.2)
        # Why is 42 used for random state?
        # A popular value chosen all over the world for the random state is 42. 
        # The reason was a little surprising and quirky. 
        # The number simply has been made popular by a comic science fiction called "The Hitchhiker's Guide to the Galaxy" 
        # authored by Douglas Adams in 1978.

        # Fit del Decision Tree depth-esimo
        dTree_clf.fit(train_x, train_y)
        print(f"training {depth}-esimo classificatore ...")

        # Calcola l'accuracy in fase di training e test con iperparametro depth-esimo
        score_train = dTree_clf.score(train_x, train_y) # accuracy training
        score_test = dTree_clf.score(test_x, test_y)    # generalization training

        # Aggiunge i valori appena calcolati alle liste, utili a stampare il plot per confronrare le curve delle due accuracy
        acc_train.append(score_train)
        acc_test.append(score_test)

    # Stampa l'acuratezza migliore in seguito al TUNING degli Iperparametri (max-depth)
    print(f"Accuratezza migliore in fase di test: {max(acc_test)}\n Iperparametro max-depth: {acc_test.index(max(acc_test))}: ")

    # ----------------  DECISION TREE with Feature Selection  ----------------
    print("\nDECISION TREE with Feature Selection")
    variances = X.var()
    thresholds = [0.2, 2, 2.3]


    for threshold in thresholds:
        #Rinizio le liste per ogni threshold
        acc_test = []
        acc_train = []
        # Seleziona le colonne con varianza maggiore di del threshold
        selected_columns = variances[variances > threshold].index.tolist()
        X_selected = X[selected_columns]
        print(f'N° Colonne selezionate: {len(selected_columns)} con threshold: {threshold}')

        for depth in max_depth_range:
            # Fit depth-esimo DecisionTreeClassifier con max_depth = depth
            dTree_clf = DecisionTreeClassifier(criterion="gini", max_depth = depth)

            train_x, test_x, train_y, test_y = train_test_split(X_selected, y, random_state = 42, test_size=0.2)

            # Fit del Decision Tree depth-esimo
            dTree_clf.fit(train_x, train_y)
            #print(f"training {depth}-esimo classificatore ...")

            # Calcola l'accuracy in fase di training e test con iperparametro depth-esimo
            score_train = dTree_clf.score(train_x, train_y) # accuracy training
            score_test = dTree_clf.score(test_x, test_y)    # generalization training

            # Aggiunge i valori appena calcolati alle liste, utili a stampare il plot per confronrare le curve delle due accuracy
            acc_train.append(score_train)
            acc_test.append(score_test)

        print(f"Accuratezza migliore in fase di test: {max(acc_test)}\n Iperparametro max-depth: {acc_test.index(max(acc_test))}: ")
        print(f"Accuracy in fase di test: {score_test}")
        print(f"Accuracy in fase di training: {score_train}\n")

    # Stampa l'acuratezza migliore in seguito al TUNING degli Iperparametri (max-depth)
   