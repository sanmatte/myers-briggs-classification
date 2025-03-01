import pandas as pd
import utils.plot as plot
import numpy as np
import cls.svm as svm
import cls.random_forest as random_forest
import cls.knn as knn
import cls.naive_bayes_custom as naive_bayes_custom
import cls.ensambe_custom as ensambe_custom
import utils.decision_tree as decision_tree
from sklearn.preprocessing import StandardScaler

def uniform_vs_zstandard(X, y):
    total_test_acc = []
    total_test_acc_std = []

    mean_of_means = X.mean().mean()
    mean_of_stds = X.std().mean()
    
    # Controlla se le medie sono vicine a 0 e le std vicine a 1
    print("\nMedia senza standardizzazione:\n", mean_of_means)
    print("Deviazione standard senza standardizzazione:\n", mean_of_stds)

    # test_acc = decision_tree.DecisionTree(X, y)
    # total_test_acc.append(test_acc)

    # test_acc = random_forest.random_forest_classifier(X, y)
    # total_test_acc_std.append(test_acc)

    # test_acc = knn.KNN_classifier(X, y, 4)
    # total_test_acc_std.append(test_acc)

    # test_acc = svm.SVM_classifier_with_tuning(X, y)
    # total_test_acc_std.append(test_acc)

    # test_acc = ensambe_custom.ensamble_classifiers(X, y)
    # total_test_acc_std.append(test_acc)
    

    # print(f"\nAccuratezza dell'albero decisionale in fase di training: \n{train_acc}")
    # print(f"Accuratezza dell'albero decisionale in fase di test: \n{test_acc}")





    # Standardizzazione
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # Calcola media e deviazione standard per ogni colonna
    mean_of_means = X_standardized.mean().mean()
    mean_of_stds = X_standardized.std().mean()

    # Controlla se le medie sono vicine a 0 e le std vicine a 1
    print("\nMedia con standardizzazione:\n", mean_of_means)
    print("Deviazione standard con standardizzazione:\n", mean_of_stds)

    # test_acc = decision_tree.DecisionTree(X_standardized, y)
    # total_test_acc.append(test_acc)

    # test_acc = random_forest.random_forest_classifier(X_standardized, y)
    # total_test_acc_std.append(test_acc)

    # test_acc = knn.KNN_classifier(X_standardized, y, 4)
    # total_test_acc_std.append(test_acc)

    # test_acc = svm.SVM_classifier_with_tuning(X_standardized, y)
    # total_test_acc_std.append(test_acc)

    # test_acc = ensambe_custom.ensamble_classifiers(X_standardized, y)
    # total_test_acc_std.append(test_acc)


    #plot.TrainingAccuracy_vs_TestAccuracy_table_standardized(total_test_acc, total_test_acc_std)

    # print(f"\nAccuratezza dell'albero decisionale in fase di training: \n{train_acc}")
    # print(f"Accuratezza dell'albero decisionale in fase di test: \n{test_acc}")
