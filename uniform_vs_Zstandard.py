import pandas as pd
import numpy as np
import decision_tree
from sklearn.preprocessing import StandardScaler


def uniform_vs_zstandard(X, y):
    mean_of_means = X.mean().mean()
    mean_of_stds = X.std().mean()
    
    # Controlla se le medie sono vicine a 0 e le std vicine a 1
    print("\nMedia senza standardizzazione:\n", mean_of_means)
    print("Deviazione standard senza standardizzazione:\n", mean_of_stds)

    train_acc, test_acc = decision_tree.DecisionTree(X, y, 15)   
    print(f"\nAccuratezza dell'albero decisionale in fase di training: \n{train_acc}")
    print(f"Accuratezza dell'albero decisionale in fase di test: \n{test_acc}")

    


    # Standardizzazione
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # Calcola media e deviazione standard per ogni colonna
    mean_of_means = X_standardized.mean().mean()
    mean_of_stds = X_standardized.std().mean()

    # Controlla se le medie sono vicine a 0 e le std vicine a 1
    print("\nMedia con standardizzazione:\n", mean_of_means)
    print("Deviazione standard con standardizzazione:\n", mean_of_stds)

    train_acc, test_acc = decision_tree.DecisionTree(X_standardized, y, 15)
    print(f"\nAccuratezza dell'albero decisionale in fase di training: \n{train_acc}")
    print(f"Accuratezza dell'albero decisionale in fase di test: \n{test_acc}")
