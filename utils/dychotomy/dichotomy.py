import pandas as pd

df = pd.read_csv("16P.csv", encoding='cp1252')
df = df.drop(columns=['Response Id'])

from sklearn.model_selection import train_test_split

# converte le personalità in dicotomie binarie
df['E/I'] = df['Personality'].apply(lambda x: 1 if x[0] == 'E' else 0)  # Extraversion (E) vs Introversion (I)
df['S/N'] = df['Personality'].apply(lambda x: 1 if x[1] == 'S' else 0)  # Sensing (S) vs Intuition (N)
df['T/F'] = df['Personality'].apply(lambda x: 1 if x[2] == 'T' else 0)  # Thinking (T) vs Feeling (F)
df['J/P'] = df['Personality'].apply(lambda x: 1 if x[3] == 'J' else 0)  # Judging (J) vs Perceiving (P)

df.drop(columns=['Personality'], inplace=True)

from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif

# domande
question_columns = df.columns[:-4]  # le ultime 4 sono E/I, S/N, T/F, J/P
print(question_columns)

correlations = {}

# calcolo delle correlazioni
for trait in ['E/I', 'S/N', 'T/F', 'J/P']:
    corr_values = []
    for question in question_columns:
        # correlazione lineare
        corr, _ = pointbiserialr(df[question], df[trait])
        corr_values.append(abs(corr))  # valore assoluto perchè ci interessa solo la forza della correlazione
    correlations[trait] = pd.Series(corr_values, index=question_columns)


corr_df = pd.DataFrame(correlations)

pd.options.display.max_rows = 60
print(corr_df.abs().sort_values(by=['E/I', 'S/N', 'T/F', 'J/P'], ascending=False))



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

X_tests = {}
y_tests = {}

def train_predict_trait(trait):
    X = df[question_columns]
    y = df[trait] 

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_tests[trait] = X_test
    y_tests[trait] = y_test

    # Training
    model = DecisionTreeClassifier(random_state=42, max_depth=12)
    model.fit(X_train, y_train)

    # valutazione
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy for {trait}: {accuracy:.2f}")

    return model

# salvo un dizionario di modelli per ogni dicotomia
models = {trait: train_predict_trait(trait) for trait in ['E/I', 'S/N', 'T/F', 'J/P']}

def predict_mbti(test_sample_df):
    mbti = ""
    for trait, model in models.items():
        prediction = model.predict(test_sample_df)[0]
        mbti += "E" if trait == "E/I" and prediction == 1 else "I" if trait == "E/I" else \
                "S" if trait == "S/N" and prediction == 1 else "N" if trait == "S/N" else \
                "T" if trait == "T/F" and prediction == 1 else "F" if trait == "T/F" else \
                "J" if prediction == 1 else "P"
    return mbti

from sklearn.metrics import accuracy_score

# valutazione del modello
def evaluate_model():
    correct_predictions = 0
    total_samples = len(X_tests['E/I'])
    print(f"Total samples: {total_samples}")
    
    predicted_types = []
    actual_types = []
    
    for i in range(total_samples):
        test_sample = X_tests['E/I'].iloc[i]  
        test_sample_df = pd.DataFrame([test_sample], columns=question_columns)

        predicted_mbti = predict_mbti(test_sample_df)

        # concatena le lettere per ottenere la personalità totale
        actual_mbti = ""
        actual_mbti += "E" if y_tests['E/I'].iloc[i] == 1 else "I"
        actual_mbti += "S" if y_tests['S/N'].iloc[i] == 1 else "N"
        actual_mbti += "T" if y_tests['T/F'].iloc[i] == 1 else "F"
        actual_mbti += "J" if y_tests['J/P'].iloc[i] == 1 else "P"

        predicted_types.append(predicted_mbti)
        actual_types.append(actual_mbti)

        if predicted_mbti == actual_mbti:
            correct_predictions += 1

    full_mbti_accuracy = correct_predictions / total_samples
    full_mbti_accuracy = accuracy_score(actual_types, predicted_types)
    print(f"Full MBTI Type Prediction Accuracy: {full_mbti_accuracy:.2%}")

    # accuracy per ogni singola dicotomia
    trait_accuracies = {
        trait: accuracy_score(y_tests[trait], models[trait].predict(X_tests[trait]))
        for trait in ['E/I', 'S/N', 'T/F', 'J/P']
    }

    print("\nTrait-wise Prediction Accuracy:")
    for trait, acc in trait_accuracies.items():
        print(f"  {trait}: {acc:.2%}")

    return full_mbti_accuracy, trait_accuracies

evaluate_model()