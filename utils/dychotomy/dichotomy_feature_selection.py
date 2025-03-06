import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ei_q = [
        "You regularly make new friends.",
        "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know",
        "You feel comfortable just walking up to someone you find interesting and striking up a conversation.",
        "You enjoy participating in group activities.",
        "After a long and exhausting week, a lively social event is just what you need.",
        "In your social circle, you are often the one who contacts your friends and initiates activities.",
        "You usually prefer to be around others rather than on your own.",
        "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places."
        ]
sn_q = [
        "You like books and movies that make you come up with your own interpretation of the ending.",
        "You are not too interested in discussing various interpretations and analyses of creative works.",
        "You become bored or lose interest when the discussion gets highly theoretical.",
        "You believe that pondering abstract philosophical questions is a waste of time.",
        "You enjoy going to art museums.",
        "You spend a lot of your free time exploring various random topics that pique your interest",
        "You are very intrigued by things labeled as controversial.",
        "You often spend a lot of time trying to understand views that are very different from your own."
        ]
tf_q = [
        "You are more inclined to follow your head than your heart.",
        "You think the world would be a better place if people relied more on rationality and less on their feelings.",
        "Seeing other people cry can easily make you feel like you want to cry too",
        "You often have a hard time understanding other people’s feelings.",
        "You know at first glance how someone is feeling.",
        "You take great care not to make people look bad, even when it is completely their fault.",
        "Your happiness comes more from helping others accomplish things than your own accomplishments.",
        "You enjoy watching people argue."
        ]
jp_q = [
        "You often make a backup plan for a backup plan.",
        "You like to use organizing tools like schedules and lists.",
        "You prefer to completely finish one project before starting another.",
        "You usually postpone finalizing decisions for as long as possible.",
        "You prefer to do your chores before allowing yourself to relax.",
        "If your plans are interrupted, your top priority is to get back on track as soon as possible.",
        "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.",
        "You complete things methodically without skipping over any steps.",
        "You often end up doing things at the last possible moment.",
        "You struggle with deadlines."
        ]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("16P.csv", encoding='cp1252')
df = df.drop(columns=['Response Id'])

# Create separate labels for each dichotomy
df['E/I'] = df['Personality'].apply(lambda x: 1 if x[0] == 'E' else 0)
df['S/N'] = df['Personality'].apply(lambda x: 1 if x[1] == 'S' else 0)
df['T/F'] = df['Personality'].apply(lambda x: 1 if x[2] == 'T' else 0)
df['J/P'] = df['Personality'].apply(lambda x: 1 if x[3] == 'J' else 0)

# Define function to train and evaluate a model for a given dichotomy
def train_dichotomy_model(questions, target):
    X = df[questions]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = DecisionTreeClassifier(max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy for {target}: {accuracy:.2%}")
    return model
# Train models for each dichotomy
models = {
    "E/I": train_dichotomy_model(ei_q, "E/I"),
    "S/N": train_dichotomy_model(sn_q, "S/N"),
    "T/F": train_dichotomy_model(tf_q, "T/F"),
    "J/P": train_dichotomy_model(jp_q, "J/P")
}

# Function to predict MBTI for a new record
def predict_mbti(X_new):
    prediction = ""
    trait_predictions = {}

    for trait, questions in zip(["E/I", "S/N", "T/F", "J/P"], [ei_q, sn_q, tf_q, jp_q]):
        input_data = pd.DataFrame([X_new[questions]], columns=questions)  # Ensure correct DataFrame format
        pred = models[trait].predict(input_data)[0]

        letter = "E" if trait == "E/I" and pred == 1 else "I" if trait == "E/I" else \
                 "S" if trait == "S/N" and pred == 1 else "N" if trait == "S/N" else \
                 "T" if trait == "T/F" and pred == 1 else "F" if trait == "T/F" else \
                 "J" if pred == 1 else "P"

        prediction += letter
        trait_predictions[trait] = letter

    return prediction, trait_predictions

# Split dataset for evaluation
X = df.drop(columns=['Personality', 'E/I', 'S/N', 'T/F', 'J/P'])  # Exclude target columns
y = df['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict and evaluate model
correct_predictions = 0
trait_correct_counts = {"E/I": 0, "S/N": 0, "T/F": 0, "J/P": 0}
total_samples = len(X_test)

for i in range(total_samples):
    test_sample = X_test.iloc[i]
    predicted_mbti, predicted_traits = predict_mbti(test_sample)

    # Get actual MBTI type
    actual_mbti = y_test.iloc[i]
    actual_traits = {
        "E/I": actual_mbti[0],
        "S/N": actual_mbti[1],
        "T/F": actual_mbti[2],
        "J/P": actual_mbti[3]
    }

    # Count correct full MBTI predictions
    if predicted_mbti == actual_mbti:
        correct_predictions += 1

    # Count correct trait-wise predictions
    for trait in ["E/I", "S/N", "T/F", "J/P"]:
        if predicted_traits[trait] == actual_traits[trait]:
            trait_correct_counts[trait] += 1

# Calculate accuracy
full_mbti_accuracy = correct_predictions / total_samples
trait_accuracies = {trait: trait_correct_counts[trait] / total_samples for trait in ["E/I", "S/N", "T/F", "J/P"]}

# Print results
print(f"\nFull MBTI Type Prediction Accuracy: {full_mbti_accuracy:.2%}")
print("\nTrait-wise Prediction Accuracy:")
for trait, acc in trait_accuracies.items():
    print(f"  {trait}: {acc:.2%}")
