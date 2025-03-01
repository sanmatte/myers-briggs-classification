import matplotlib.pyplot as mpl
import seaborn as sns

# Mostra la matrice di correlazione fra le diverse domande

def correlation_matrix_chart(df):

    # Calcolo della matrice di correlazione
    correlation_matrix = df.corr()

    # Creazione della figura
    mpl.figure(figsize=(8, 7))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)

    # Mostra il grafico
    mpl.title("Matrice di Correlazione", fontsize = 18)
    mpl.show()

# Mostra la distribuzione del dataset - Evidenzia la distribuzione ≈ Uniforme
def distribution_chart(df):

    # Specifica il numero di bins == 16 (personalità)
    df['Personality'].hist(bins=16)

    # Aggiunge il titolo e le etichette
    mpl.xlabel('Personalities')
    mpl.xticks(rotation=45)
    mpl.ylabel('Frequency')
    mpl.title('Distribution of Personalities')
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