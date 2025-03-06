from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

def compute_accuracy(pred_y, test_y):
    return (pred_y == test_y).sum() / len(pred_y)

# oversampling
def smote(df):
    # campiona dataset
    df_sampled = df.sample(frac=0.5, random_state=42)

    X = df_sampled.drop(columns=['Personality', 'Response Id'], axis=1) 
    y = df_sampled['Personality']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(train_x, train_y)
    for cls, count in Counter(y_resampled).items():
        print(f"{cls} : {count}")

    # Knn
    knn = KNeighborsClassifier(4)
    knn.fit(X_resampled, y_resampled)
    pred_y = knn.predict(test_x)
    print('Accuratezza del knn dopo SMOTE oversampling %s' % compute_accuracy(test_y, pred_y))

def random_over(df):
    # campiona dataset
    df_sampled = df.sample(frac=0.5, random_state=42)

    X = df_sampled.drop(columns=['Personality', 'Response Id'], axis=1) 
    y = df_sampled['Personality']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)
    rus = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(train_x, train_y)
    for cls, count in Counter(y_resampled).items():
        print(f"{cls} : {count}")

    # Knn
    knn = KNeighborsClassifier(4)
    knn.fit(X_resampled, y_resampled)
    pred_y = knn.predict(test_x)
    print('Accuratezza del knn dopo random oversampling %s' % compute_accuracy(test_y, pred_y))


# undersampling

def cluster_centroids(df):
    # campiona dataset
    df_sampled = df.sample(frac=0.5, random_state=42)

    X = df_sampled.drop(columns=['Personality', 'Response Id'], axis=1) 
    y = df_sampled['Personality']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)
    cc = ClusterCentroids(random_state=42)
    X_resampled, y_resampled = cc.fit_resample(train_x, train_y)
    for cls, count in Counter(y_resampled).items():
        print(f"{cls} : {count}")

    # Knn
    knn = KNeighborsClassifier(4)
    knn.fit(X_resampled, y_resampled)
    pred_y = knn.predict(test_x)
    print('Accuratezza del knn dopo cluster centroids %s' % compute_accuracy(test_y, pred_y))

def random_under(df):
    # campiona dataset
    df_sampled = df.sample(frac=0.5, random_state=42)

    X = df_sampled.drop(columns=['Personality', 'Response Id'], axis=1) 
    y = df_sampled['Personality']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(train_x, train_y)
    for cls, count in Counter(y_resampled).items():
        print(f"{cls} : {count}")

    # Knn
    knn = KNeighborsClassifier(4)
    knn.fit(X_resampled, y_resampled)
    pred_y = knn.predict(test_x)
    print('Accuratezza del knn dopo random undersampling %s' % compute_accuracy(test_y, pred_y))



