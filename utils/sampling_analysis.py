from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_test_model(train_x, train_y, test_x, test_y):
    model_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=16, min_samples_split=2, min_samples_leaf=1)

    model_clf.fit(train_x, train_y)
    pred_y = model_clf.predict(test_x)
    return accuracy_score(test_y, pred_y)

def sampling_analysis(X, y):
    dim_range = []
    model_acc = []
    model_acc_strat = []

    # esegue un primo split con stratificazione per essere sicuro che il test set sia rappresentativo
    sample_x, test_x, sample_y, test_y = train_test_split(X, y, random_state=42, test_size=0.20, stratify=y)
    orig_train_size = np.shape(sample_y)[0]
    dim_range = dim_range + [orig_train_size]

    # testo l'intero train set per il model e lo memorizzo in entrambe le liste
    acc = train_test_model(sample_x, sample_y, test_x, test_y)
    model_acc = model_acc + [acc]
    model_acc_strat = model_acc_strat + [acc]

    # crea un range da 0.8 a 0.1 (differente da range perchè crea interalli equidistanti)
    train_sizes = np.linspace(0.8, 0.1, num=8)

    for train_size in train_sizes:

        #split senza stratificazione
        train_x, _, train_y, _ = train_test_split(sample_x, sample_y, random_state=42, train_size=train_size)
        dim_range.append(dim_range[0]*train_size)
        #print(dim_range)
        u = train_test_model(train_x, train_y, test_x, test_y)
        print(f"Without stratify\nAccuracy: {u} with train_size = {train_size}\n")
        model_acc = model_acc + [u]
        
        #split con stratificazione
        train_x, _, train_y, _ = train_test_split(sample_x, sample_y, random_state=42, train_size=train_size, stratify=sample_y)
        y = train_test_model(train_x, train_y, test_x, test_y)
        print(f"With stratify\nAccuracy: {u} with train_size = {train_size}\n")
        model_acc_strat = model_acc_strat + [y]
    
        print(f"With Startify\nMax accuracy: {max(model_acc_strat)} index {model_acc_strat.index(max(model_acc_strat))}\nMin accuracy: {min(model_acc_strat)}\nDifference: {max(model_acc_strat)-min(model_acc_strat)}")


        plt.ylim([0.9595, 0.9785])
        plt.yticks(np.arange(0.9595, 0.9785, 0.002))  # Tick ogni 0.002
        plt.ticklabel_format(style='plain', axis='y')  # Evita la notazione scientifica

        # Imposta i limiti e i tick per X (data size)
        plt.xlim([min(dim_range), max(dim_range)])

        # Grafico di confronto delle accuracy al variare della data size
        plt.plot(dim_range, model_acc, lw=2, label=['model random'], marker='o')
        plt.plot(dim_range, model_acc_strat, lw=2, label=['model stratified'], marker='s')
        plt.grid(True, axis='both', zorder=0, linestyle=':', color='k')
        plt.tick_params(labelsize=12)
        plt.xlabel('data size', fontsize=24)
        plt.ylabel('Accuracy', fontsize=24)
        plt.title('Model Performances', fontsize=24)
        plt.legend()
        plt.show()
    