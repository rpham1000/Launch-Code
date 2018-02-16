import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

# Import data that has been processed in R
train_df = pd.DataFrame.from_csv('train_cleaned.csv')
train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
train_df['Age'].fillna(200, inplace=True)
# train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)

# Save for Kaggle Submission
# test_df = pd.read_csv('test_cleaned.csv')

features = train_df.drop(
    ['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
features = features.values.tolist()
labels = train_df['Survived']
labels = labels.values.tolist()

acc = []
# clf = tree.DecisionTreeClassifier()
clf = AdaBoostClassifier()
for i in range(1, 100):
    # Split the training data
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.10, random_state=i)

    # Classification
    # print("Training...")
    clf = clf.fit(features_train, labels_train)
    # print("Trained")

    acc_trial = clf.score(features_test, labels_test)
    acc.append(acc_trial)

print("Accuracy: {}".format(np.mean(acc)))
