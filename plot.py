import matplotlib.pyplot as mpl
import numpy as np
import seaborn as sns

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
    