import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix


def train():
    data = pd.read_csv('fake_or_real_news.csv')
    data = data.drop('Unnamed: 0', axis=1)

    le = LabelEncoder()
    data.iloc[:, 2] = le.fit_transform(data.iloc[:, 2])

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    cv = CountVectorizer(max_features=5000)
    mat_body = cv.fit_transform(X[:, 1]).todense()

    cv_head = CountVectorizer(max_features=5000)
    mat_head = cv_head.fit_transform(X[:, 0]).todense()

    X_mat = np.hstack((mat_head, mat_body))
    print(X_mat[-1])
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y, test_size=0.2, random_state=0)
    dtc = DecisionTreeClassifier(criterion='entropy')
    dtc.fit(X_train, y_train)

    filename = 'model/news_model1.sav'
    pickle.dump(dtc, open(filename, 'wb'))

    y_pred = dtc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
