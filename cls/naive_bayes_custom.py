import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer

start = timer()
# ...




def get_class_proba(y):
    """
    Calcola la probabilità di appartenenza ad ogni classe.

    Args:
        y (np.array): Array di classi.

    Returns:
        class_probs (dict): Dizionario con le probabilità di appartenenza ad ogni classe.
    """
    n = len(y)
    y_vals, y_counts = np.unique(y, return_counts=True)
    class_probs = {y_val: y_count/n for y_val, y_count in zip(y_vals, y_counts)} # {'CLASSE' : P(CLASSE)}

    return class_probs
        

# def get_att_proba(attribute,  y):
#     """
#     Calcola la probabilità di condizionata di appartenenza alle classi dato il valore dell'attributo.

#     Args:
#         attribute (np.array): Array di valori dell'attributo.
#         y (np.array): Array di classi.
    
#     Returns:
#         probs_x_att (dict): Dizionario con le probabilità di appartenenza per ogni valore dell'attributo.
#     """
#     y_vals, y_counts = np.unique(y, return_counts=True)

#     probs_x_att = probs_x_att = {y_val: [0] * 7 for y_val in y_vals}
#     value_to_index = { -3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6 } # mappa valori del dataset a indici fissi
    
#     for i, y_val in enumerate(y_vals):
#         a_vals, a_counts = np.unique(attribute, return_counts=True)
#         for j, a_val in enumerate(a_vals):
#             count = sum(np.array(y==y_val) & np.array(attribute==a_val))
#             probs_x_att[y_val][value_to_index[a_val]] = count/y_counts[i] # {CLASSE: [P(-3|CLASSE), P(-2|CLASSE), ..., P(3|CLASSE)]}

#     return probs_x_att

def get_att_proba(attribute, y):
    """
    Calcola la media e la deviazione standard per ogni valore dell'attributo condizionato alla classe.

    Args:
        attribute (np.array): Array di valori dell'attributo.
        y (np.array): Array di classi.
    
    Returns:
        probs_x_att (dict): Dizionario con media e deviazione standard per ogni classe.
    """
    y_vals = np.unique(y)
    probs_x_att = {}

    for y_val in y_vals:
        attr_given_class = attribute[y == y_val]  # Filtra valori dell'attributo per la classe
        mean = np.mean(attr_given_class) # media
        std = np.std(attr_given_class)   # deviazione standard
        probs_x_att[y_val] = (mean, std)

    return probs_x_att

def predict(X_test, y_test, class_probs, entire_probability):
    """
    Predice la classe di appartenenza di un'istanza.

    Args:
        x (np.array): Istanza da classificare.
        y (np.array): Array di classi.
        class_probs (dict): Probabilità di appartenenza alle classi.
        entire_probability (dict): Probabilità condizionata per ogni attributo.

    Returns:
        max_class (int): Classe predetta.
    """
    y_pred = []
    # values_to_index = { -3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6 } # mappa valori del dataset a indici fissi
    attr_list = X_test.columns.tolist()

    for i in range(0, np.shape(X_test)[0]):
        max_prob = -1
        max_class = None
        for y_val in class_probs.keys():
            prob = class_probs[y_val]
            for j in range(0, np.shape(X_test)[1]):
                mean, std = entire_probability[attr_list[j]][y_val]
                prob *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((X_test.iloc[i, j] - mean) ** 2) / (2 * std ** 2)) # calcola la probabilità di appartenenza ad una classe con la formmula di ripartizione
                # prob *= entire_probability[attr_list[j]][y_val][values_to_index[X_test.iloc[i,j]]] # calcola la probabilità di appartenenza ad una classe: P(CLASSE) * P(ATTRIBUTO1|CLASSE) * P(ATTRIBUTO2|CLASSE) * ... * P(ATTRIBUTOn|CLASSE)
            if prob > max_prob:
                max_prob = prob
                max_class = y_val
        y_pred.append(max_class)

    accuracy = np.mean(y_pred == y_test)
    return accuracy
    


import numpy as np

def Naive_Bayes_Custom(X, y):
    """
    Implementa il classificatore Naive Bayes e stampa 

    Args:
        X (pd.DataFrame): Dataset di attributi.
        y (pd.Series): Serie di classi.
    """
    print("Starting Naive Bayes Custom")
    start = timer()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sub_start = timer()
    print("Calcolo probabilità di appartenenza alle classi: ", end="")
    class_probs = get_class_proba(y_train)
    sub_end = timer()
    print(f"{(sub_end - sub_start):.2f}, seconds")
    question_columns = X.columns.tolist() # lista di attributi
    entire_probability = {} # dizionario con probabilità di appartenenza per ogni attributo

    sub_start = timer()
    print("Calcolo probabilità condizionata per ogni attributo: ", end="")
    for col in range(0,np.shape(X_train)[1]):
        entire_probability.update({question_columns[col]: get_att_proba(X_train.iloc[:,col], y_train)})
    sub_end = timer()
    print(f"{(sub_end - sub_start):.2f}, seconds")

    train_accuracy = "Not Calculated to reduce complexity time"
    # train_accuracy = predict(X_train, y_train, class_probs, entire_probability)
    test_accuracy = predict(X_test, y_test, class_probs, entire_probability)
    end = timer()
    ctime = end - start

    ctime, metric = [ctime, "seconds"] if ctime < 120 else [ctime/60, "minutes"]
    print(f"Done: {ctime:.2f}, {metric}\n")
    
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    return train_accuracy, test_accuracy
