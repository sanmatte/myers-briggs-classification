import matplotlib.pyplot as mpl
import numpy as np
import seaborn as sns
from plotly.offline import iplot
import plotly.express as px
import pandas as pd

# Mostra la matrice di correlazione fra le diverse domande
def correlation_matrix_chart(X, corr_method):

    # Calcolo della matrice di correlazione
    correlation_matrix = X.corr(method = corr_method)

    # Creazione della figura
    mpl.figure(figsize=(8, 7))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)

    # Mostra il grafico
    mpl.title(f"Matrice di Correlazione - Coefficiente di correlazione: {corr_method}", fontsize = 18)
    mpl.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_likert(X):

    # Definizione delle categorie di risposta
    response_mapping = {
        -3: "Molto in disaccordo",
        -2: "Disaccordo",
        -1: "Abbastanza in disaccordo",
        0: "Né d’accordo né in disaccordo",
        1: "Abbastanza d’accordo",
        2: "D’accordo",
        3: "Molto d’accordo"
    }

    # Rinomina le colonne con un identificativo progressivo Q# al posto dell'intera domanda per una maggiore leggibilità
    qq = ['Q'+str(i) for i in range(1,len(X.columns)+1)]
    X.columns = qq

    # Colori per le categorie Likert
    colors = ["firebrick", "lightcoral", "rosybrown", "gainsboro", "lightskyblue", "cornflowerblue", "darkblue"]

    # Seleziona casualmente 5 colonne
    random_columns = X.sample(n=8, axis=1).columns

    # Filtra il DataFrame con le colonne scelte
    data_subset = X[random_columns]

    # Conta le occorrenze di ogni risposta per ogni domanda selezionata
    response_counts = data_subset.apply(pd.Series.value_counts).fillna(0).astype(int)

    # Rinomina le righe con le etichette di risposta
    response_counts.index = response_counts.index.map(response_mapping)

    # Ordina le risposte secondo la scala Likert
    response_counts = response_counts.loc[["Molto in disaccordo", "Disaccordo", "Abbastanza in disaccordo",
                                        "Né d’accordo né in disaccordo", "Abbastanza d’accordo",
                                        "D’accordo", "Molto d’accordo"]]

    # Trasponi per avere domande come righe e categorie Likert come colonne
    likert_data = response_counts.T

    # Converti in percentuali
    likert_data_percentage = likert_data.div(likert_data.sum(axis=1), axis=0) * 100

    # Creazione del grafico
    fig, ax = plt.subplots(figsize=(12, 8))
    cumsum_data = likert_data_percentage.cumsum(axis=1)

    for i, category in enumerate(likert_data_percentage.columns):
        ax.barh(likert_data_percentage.index, likert_data_percentage[category], 
                left=(cumsum_data[category] - likert_data_percentage[category]), 
                color=colors[i], label=category)

    ax.set_xlabel("Percentage")
    ax.set_title("Distribuzione delle risposte Likert")
    ax.legend(title="Scala Likert", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Linea verticale per evidenziare il neutrale
    ax.axvline(50, color="gray", linestyle="dashed")

    plt.tight_layout()
    plt.show()

def question_distribution_chart(X):
    hex_colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080",
    "#008000", "#000080", "#800000", "#808000", "#008080", "#C0C0C0", "#808080", "#FFC0CB",
    "#800000", "#A52A2A", "#800080", "#008000", "#000080", "#708090", "#2F4F4F", "#D2691E",
    "#B22222", "#228B22", "#20B2AA", "#4682B4", "#F0E68C", "#FF4500", "#FFD700", "#ADFF2F",
    "#8A2BE2", "#CD5C5C", "#FF6347", "#40E0D0", "#7FFF00", "#DAA520", "#B0C4DE", "#87CEEB",
    "#778899", "#6A5ACD", "#FF69B4", "#20B2AA", "#F5DEB3", "#F0FFFF", "#32CD32", "#FFB6C1",
    "#9932CC", "#FF8C00", "#87CEFA", "#E9967A", "#00CED1", "#DDA0DD", "#8B008B", "#556B2F",
    "#BDB76B", "#ADFF2F", "#4B0082", "#800000", "#CD853F"
    ]

    qq = ['Q'+str(i) for i in range(1,len(X.columns)+1)]
    X.columns = qq

    e = 0
    for i in qq:
        iplot(px.histogram(X,x=i,color_discrete_sequence=[hex_colors[e]],text_auto=True))
        e = e + 1

# Mostra la distribuzione del dataset - Evidenzia la distribuzione ≈ Uniforme
def distribution_chart(df):

    # Specifica il numero di bins == 16 (personalità)
    #df['Personality'].hist(bins=16)
    df.hist(bins=16)

    # Aggiunge il titolo e le etichette
    mpl.xlabel('Personalities')
    mpl.xticks(rotation=45)
    mpl.ylabel('Frequency')
    mpl.title('Distribution of Personalities - Training Set')
    mpl.show()

# Confronta l'accuratezza dei modelli addestrati sul training set e sul test set
def trainingAccuracy_vs_testAccuracy_chart(x_scale, y_scale, acc_train, acc_test, curve1_label, curve2_label, x_label, y_label, plot_title):
    mpl.figure(figsize=(8, 5))
    mpl.plot(x_scale, acc_train, label = curve1_label, lw=2, marker='o')
    mpl.plot(y_scale, acc_test, label = curve2_label, lw=2, marker='s')
    mpl.grid()
    mpl.xlabel(x_label, fontsize = 15)
    mpl.ylabel(y_label, fontsize = 15)
    mpl.title(plot_title, fontsize = 22)
    mpl.legend()
    mpl.show()

def TrainingAccuracy_vs_TestAccuracy_standardized(acc_train, acc_test):
    # Indici per i modelli
    labels = []
    labels = ["Decision Tree", "Random Forest", "kNN",  "SVM", "Ensemble"]

    # Creazione del grafico a barre orizzontali
    mpl.figure(figsize=(8, 6))
    y_pos = np.arange(len(labels))

    bar_width = 0.4  # Larghezza delle barre
    mpl.barh(y_pos - bar_width/2, acc_train, height=bar_width, label='Test', color='#1f77b4')
    mpl.barh(y_pos + bar_width/2, acc_test, height=bar_width, label='Test standardizzato', color='#ff7f0e')

    # Etichette e titolo
    mpl.xlabel("Accuratezza (%)", fontsize = 14)
    mpl.ylabel("Modelli a confronto", fontsize = 14)
    mpl.title("Confronto delle Acuratezze dei modelli con e senza Z-score", fontsize = 18)
    mpl.yticks(y_pos, labels)
    mpl.xlim(60, 100)  # Impostiamo i limiti dell'asse x
    mpl.grid(True, linestyle='--', alpha=0.6, axis='x')
    mpl.legend()

    # Mostra il grafico
    mpl.show()
    
