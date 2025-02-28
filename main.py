import plot
import knn_features
import ourTest
import decision_tree
import knn
import pandas as pd
import numpy as np
from simple_term_menu import TerminalMenu


def start_menu():

    main_options = [" [1] Data Analysis", " [2] Classificatori a confronto", " [3] Fai il test", " [4] Quit"]
    classifier_options = ["  [a] Decision-Tree (Scikit-learn)", "  [b] kNN (Scikit-learn)", "  [c] kNN con features(Scikit-learn)", "  [d] Decison-Tree (Custom)", "  [e] kNN (Custom)", "  [f] Return to Main Menu"]
    verbose_options = [" [a] Alto", " [b] Basso"]

    mainMenu = TerminalMenu(main_options, title = "MAIN MENU")
    classifierMenu = TerminalMenu(classifier_options, title = "CLASSIFICATORI A CONFRONTO") 
    verboseMenu = TerminalMenu(verbose_options, title = "LIVELLO DI DETTAGLIO")

    quitting = False

    #----------- MAIN MENU ----------#
    while quitting == False:
        optionsIndex = mainMenu.show()
        optionsChoice = main_options[optionsIndex]

        # '[1] Data Analysis'
        if optionsChoice == main_options[0]:
            verboseIndex = verboseMenu.show()
            verboseChoice = verbose_options[verboseIndex]

            if verboseChoice == " [a] Alto":
                # Fornisce informazioni generali sul dataset (Summary)
                print(df.describe().T)

            if verboseChoice == " [b] Basso":
                ...

        # '[2] Classificatori a confronto'
        if optionsChoice ==  main_options[1]:
            returnToMainMenu = False
            while(returnToMainMenu == False):
                classifierIndex = classifierMenu.show()
                classifierChoice = classifier_options[classifierIndex]

                if classifierChoice == "  [a] Decision-Tree (Scikit-learn)":
                    decision_tree.DecisionTree_with_tuning(X, y)

                if classifierChoice == "  [b] kNN (Scikit-learn)":
                    knn.KNN_classifier_with_tuning(X, y)

                if classifierChoice == classifier_options[2]:
                    knn_features.kNN_classifier_with_Feature_Selection(X, y)
                
                if classifierChoice == "[d] Decison-Tree (Custom)":
                    ...

                if classifierChoice ==  "  [e] kNN (Custom)":
                    ...

                if classifierChoice == "  [f] Return to Main Menu":
                    returnToMainMenu = True

        # '[3] Fai il test'
        if optionsChoice ==  main_options[2]:
            ourTest.start_survey()

        # '[4] Quit' 
        if optionsChoice ==  main_options[3]:
            quitting = True


if __name__ == "__main__":

    # Legge il file csv dal path specificato e lo inserisce in un dataframe pandas
    df = pd.read_csv('16P.csv', encoding='cp1252') # 0x92 is usually a smart quote in the windows-1252 encoding.It is not a valid UTF-8 character, so that's why csv refuses to parse it. (from reddit)

    # Divide il dataframe in data-matrix (X) e label-vector (y)
    # X = df.iloc[:, 1:(len(df.columns) - 1)]
    # y = df.iloc[:, len(df.columns)-1:]
    X = df.drop(['Personality', 'Response Id'], axis=1)
    y = df['Personality']
    pd.options.display.max_rows = 5000                 

    # Menu Principale di scelta
    start_menu()