import ourTest
import plot
import knn
import knn_features
import random_forest
import naive_bayes_custom
import svm
import decision_tree
from copy import deepcopy
import ensambe_custom
import pandas as pd
import numpy as np
from simple_term_menu import TerminalMenu

def data_analysis(df):
    
     # Calcola e stampa la MATRICE di CORRELAZIONE
    print('\n------------------------------------------------------------------------------------------------------------------------------------')
    print('\nMATRICE di CORRELAZIONE\n')
    # Rinomina le intestazioni delle domande con un identificativo progessivo Q# al posto della domanda intera
    tmp_df = deepcopy(df)
    qq = ['Q'+str(i) for i in range(1,len(tmp_df.columns)+1)]
    tmp_df.columns = qq
    print(tmp_df.corr().T)

    # Fornisce delle STATISTICHE GENERALI riguardo il dataset
    print('\n------------------------------------------------------------------------------------------------------------------------------------')
    print('\nSTATISTICHE GENERALI\n')
    print(df.describe().T)

    # Fornisce la tabella delle FREQUENZE ASSOLUTE delle classi
    print('\n------------------------------------------------------------------------------------------------------------------------------------')
    print('\nFREQUENZE ASSOLUTE\n')
    print(df.groupby('Personality').agg({'Personality': 'count'}))
    print('\n------------------------------------------------------------------------------------------------------------------------------------')

def start_menu():

    # Liste delle opzioni per i diversi menù e sotto-menù
    main_options = [" [1] Data Analysis", " [2] Pre-processing", " [3] Classificatori", " [4] Fai il test", " [5] Quit"]
    classifier_options = [" [a] kNN (Scikit-learn)", " [b] Random Forest(Scikit-learn)", " [c] SVM (Scikit-learn)", " [d] Naive-Bayes (Custom)", " [e] Ensemble (Custom)", " [f] Return to Main Menu"]
    pre_pro_options = [" [a] Attiva Feature Selection", " [b] Disattiva Feature Selection", " [c] Simula Undersampling",]

    # Istanze dei menù
    mainMenu = TerminalMenu(main_options, title = "MAIN MENU")
    classifierMenu = TerminalMenu(classifier_options, title = "CLASSIFICATORI A CONFRONTO") 

    # Flag principale impiegato nel loop del menù principale
    quitting = False

    #----------- MAIN MENU ----------#
    while quitting == False:
        optionsIndex = mainMenu.show()
        optionsChoice = main_options[optionsIndex]

        # >>> [1] Data Analysis
        if optionsChoice == main_options[0]:
            data_analysis(df)

        # >>> [2] Pre-processing
        if optionsChoice ==  main_options[1]:
            ...

        # >>> [3] Classificatori
        if optionsChoice ==  main_options[2]:
            returnToMainMenu = False
            while(returnToMainMenu == False):
                classifierIndex = classifierMenu.show()
                classifierChoice = classifier_options[classifierIndex]
                
                # [a] kNN (Scikit-learn)
                if classifierChoice == classifier_options[0]:
                    knn.KNN_classifier_with_tuning(X, y)

                # [b] Random Forest(Scikit-learn)
                if classifierChoice == classifier_options[1]:
                    random_forest.random_forest_classifier(X, y)

                # [c] SVM (Scikit-learn)
                if classifierChoice == classifier_options[2]:
                    svm.SVM_classifier_with_tuning(X, y)

                # [d] Naive-Bayes (Custom)
                if classifierChoice == classifier_options[3]:
                    naive_bayes_custom.Naive_Bayes_Custom(X, y)

                # [e] Ensemble (Custom)
                if classifierChoice ==  classifier_options[4]:
                    ensambe_custom.ensamble_classifiers(X, y)

                # [f] Return to Main Menu
                if classifierChoice == classifier_options[5]:
                    returnToMainMenu = True

        # >>> [4] Fai il test
        if optionsChoice ==  main_options[3]:
            ourTest.start_survey()

        # >>> [5] Quit
        if optionsChoice ==  main_options[4]:
            quitting = True


if __name__ == "__main__":

    # Legge il file csv dal path specificato e lo inserisce in un dataframe pandas
    df = pd.read_csv('16P.csv', encoding='cp1252') # 0x92 is usually a smart quote in the windows-1252 encoding.It is not a valid UTF-8 character, so that's why csv refuses to parse it. (from reddit)

    # Divide il dataframe in data-matrix (X) e label-vector (y)
    X = df.drop(['Personality', 'Response Id'], axis=1)
    y = df['Personality']

    # Consente la stampa estesa sul terminale
    pd.options.display.max_rows = 5000                 

    # Menu Principale di scelta
    start_menu()