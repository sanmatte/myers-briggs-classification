import test.ourTest as ourTest
import utils.plot as plot
import cls.knn as knn
import preprocessing.feature_selection as feature_selection
import cls.random_forest as random_forest
import cls.naive_bayes_custom as naive_bayes_custom
import cls.svm as svm
import utils.decision_tree as decision_tree
import utils.uniform_vs_Zstandard as uniform_vs_Zstandard
from copy import deepcopy
import cls.ensambe_custom as ensambe_custom
import pandas as pd
import numpy as np
from simple_term_menu import TerminalMenu

def data_analysis(X):
    
     # Calcola e stampa la MATRICE di CORRELAZIONE
    print('\n------------------------------------------------------------------------------------------------------------------------------------')
    print('\nMATRICE di CORRELAZIONE\n')
    
    # Rinomina le intestazioni delle domande con un identificativo progessivo Q# al posto dell' intera domanda per una maggiore leggibilità
    tmp_X = deepcopy(X)
    qq = ['Q'+str(i) for i in range(1,len(tmp_X.columns)+1)]
    tmp_X.columns = qq
    corr_method = "kendall"
    print(tmp_X.corr(method = corr_method))

    # Se vuoi visualizzare il plot della matrice di correlazione, scommenta questa riga
    plot.correlation_matrix_chart(tmp_X, corr_method)

    # Fornisce delle STATISTICHE GENERALI riguardo il dataset
    print('\n------------------------------------------------------------------------------------------------------------------------------------')
    print('\nSTATISTICHE GENERALI\n')
    print(df.describe().T)

    # Fornisce la tabella delle FREQUENZE ASSOLUTE delle classi
    print('\n------------------------------------------------------------------------------------------------------------------------------------')
    print('\nFREQUENZE ASSOLUTE\n')
    print(df.groupby('Personality').agg({'Personality': 'count'}))
    print('\n------------------------------------------------------------------------------------------------------------------------------------')

FEATURES_SWITCH = {0: 'OFF', 1: 'ON'}
switch_value = 0

def start_menu():
    global switch_value
    # Liste delle opzioni per i diversi menù e sotto-menù
    main_options = [" [1] Data Analysis", " [2] Pre-processing", " [3] Classificatori", " [4] Fai il test", " [0] Esci"]
    classifier_options = [" [3.1] kNN (Scikit-learn)", " [3.2] Random Forest(Scikit-learn)", " [3.3] SVM (Scikit-learn)", " [3.4] Naive-Bayes (Custom)", " [3.5] Ensemble (Custom)", " [0] Indietro"]
    pre_pro_options = [f" [2.1] Feature Selection: {FEATURES_SWITCH[switch_value]}", " [2.2] Undersampling", " [2.3] Oversampling", " [2.4] Standardizzazione", " [0] Indietro"]
    back_exit_options = [" [0] Indietro", " [1] Esci"]

    # Istanze dei menù
    mainMenu = TerminalMenu(main_options, title = "MAIN MENU")
    classifierMenu = TerminalMenu(classifier_options, title = "CLASSIFICATORI A CONFRONTO")
    back_exit_menu = TerminalMenu(back_exit_options, title = "OPZIONI")

    # Flag principale impiegato nel loop del menù principale
    quitting = False

    #----------- MAIN MENU ----------#
    while quitting == False:
        optionsIndex = mainMenu.show()
        optionsChoice = main_options[optionsIndex]

        # >>> [1] Data Analysis
        if optionsChoice == main_options[0]:
            data_analysis(X)

        # >>> [2] Pre-processing
        if optionsChoice ==  main_options[1]:
            exit_pre_processing_settings = False
            while exit_pre_processing_settings == False:
                pre_processing_index = TerminalMenu(pre_pro_options, title = "PRE-PROCESSING").show()
                pre_pro_choice = pre_pro_options[pre_processing_index]
                if pre_pro_choice == pre_pro_options[0]:
                    switch_value = 1 - switch_value
                    pre_pro_options[0] = f" [a] Feature Selection: {FEATURES_SWITCH[switch_value]}"
                if pre_pro_choice == pre_pro_options[1]:
                    plot.plot_undersampling(X, y)
                if pre_pro_choice == pre_pro_options[2]:
                    plot.plot_oversampling(X, y)
                if pre_pro_choice == pre_pro_options[3]:
                    uniform_vs_Zstandard.uniform_vs_zstandard(X, y)
                if pre_pro_choice == pre_pro_options[4]:
                    exit_pre_processing_settings = True


        # >>> [3] Classificatori
        if optionsChoice ==  main_options[2]:
            classifiers_array = []
            if switch_value == 0:
                classifiers_array = [knn.KNeighborsClassifier, random_forest.random_forest_classifier, svm.SVM_classifier, naive_bayes_custom.Naive_Bayes_Custom, ensambe_custom.ensamble_classifiers]
            else:
                threshold = 2.0
                variances = X.var()
                X_selected = X[variances[variances > threshold].index.tolist()]
                #classifiers_array = [knn_features.kNN_classifier_with_Feature_Selection, random_forest.random_forest_classifier, svm.SVM_classifier_with_tuning, naive_bayes_custom.Naive_Bayes_Custom, ensambe_custom.ensamble_classifiers]
                #! TO FIX: implement feature selection for all classifiers
                
            returnToMainMenu = False
            while(returnToMainMenu == False):
                classifierIndex = classifierMenu.show()
                classifierChoice = classifier_options[classifierIndex]
                
                # [a] kNN (Scikit-learn)
                if classifierChoice == classifier_options[0]:
                    if switch_value == 0:
                        # remove this to tune the model yourself
                        # knn.KNN_classifier_with_tuning(X, y)
                        knn.KNN_classifier(X, y)
                    else:
                        # remove this to calculate yourself the best threshold
                        threshold, X_selected = feature_selection.better_threshold(X, y, knn.KNN_classifier, k=4)
                        knn.KNN_classifier(X_selected, y)
                    

                # [b] Random Forest(Scikit-learn)
                if classifierChoice == classifier_options[1]:
                    classifiers_array[1](X, y)

                # [c] SVM (Scikit-learn)
                if classifierChoice == classifier_options[2]:
                    classifiers_array[2](X, y)


                ##### ! TO FIX: SOME MENU LOGIC IS MISSING HERE #####
                ##### ! TO FIX: SOME MENU LOGIC IS MISSING HERE #####
                ##### ! TO FIX: SOME MENU LOGIC IS MISSING HERE #####
                ##### ! TO FIX: SOME MENU LOGIC IS MISSING HERE #####
                ##### ! TO FIX: SOME MENU LOGIC IS MISSING HERE #####
                # [d] Naive-Bayes (Custom)
                if classifierChoice == classifier_options[3]:
                    classifiers_array[3](X, y)
                    r = back_exit_menu.show()
                    if r == 1:
                        returnToMainMenu = True

                # [e] Ensemble (Custom)
                if classifierChoice ==  classifier_options[4]:
                    classifiers_array[4](X, y)

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
    