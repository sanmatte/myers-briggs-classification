import matplotlib.pyplot as mpl

# Mostra la distribuzione del dataset - Evidenzia la distribuzione ≈ Uniforme
def distribution_chart(df):

    # Specifica il numnero di bins == 16 (personalità)
    df['Personality'].hist(bins=16)

    # Aggiunge il titolo e le etichette
    mpl.xlabel('Personalities')
    mpl.xticks(rotation=45)
    mpl.ylabel('Frequency')
    mpl.title('Distribution of Personalities')
    mpl.show()

# Confronta l'accuratezza dei modelli addestrati sul training set e sul test set
def trainingAccuracy_vs_testAccuracy_chart(max_depth_range, acc_train, acc_test):
    mpl.plot(max_depth_range, acc_train, lw=2, color='r')
    mpl.plot(max_depth_range, acc_test, lw=2, color='b')
    mpl.xlim([1, max(max_depth_range)])
    mpl.grid(True, axis = 'both', zorder = 0, linestyle = ':', color = 'k')
    mpl.tick_params(labelsize = 18)
    mpl.xlabel('max_depth', fontsize = 24)
    mpl.ylabel('Accuracy', fontsize = 24)
    mpl.title('Model Performances', fontsize = 24)
    mpl.legend(['Train', 'Test'])
    mpl.show()