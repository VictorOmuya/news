from re import I
from flask import Flask, render_template, redirect, url_for, request
from numpy.lib.function_base import vectorize
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


app = Flask(__name__)


@app.route('/', methods=['GET'])
def begin():

    return render_template('index.html', msg='')


@app.route('/preds', methods=['GET', 'POST'])
def det():
    if request.method == 'POST':
        result = request.form

        body = result['body']

        loaded_model = pickle.load(open('model/fake_model.pkl', 'rb'))
        tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
        dataframe = pd.read_csv('fake_or_real_news.csv')
        x = dataframe['text']
        y = dataframe['label']
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0)

        tfid_xtrain = tfvect.fit_transform(x_train)
        tfid_xtest = tfvect.transform(x_test)

        input_text = [body]
        vectorized_input = tfvect.transform(input_text)

        prediction = loaded_model.predict(vectorized_input)
        print(prediction)
        result = ""
        for a in prediction:
            a = str(a)
            if a == 'FAKE':
                result = 'Fake'
            elif a == 'REAL':
                result = 'Real'
            return render_template('detect.html', result=result)

    return render_template('detect.html')


@app.route('/chek', methods=['GET', 'POST'])
def chk():
    if request.method == 'POST':
        result = request.form
        body = result['body']
        head = result['headline']
        loaded_model = pickle.load(open('model/news_model.sav', 'rb'))

        data = pd.read_csv('fake_or_real_news.csv')
        data = data.drop(["Unnamed: 0", "label"], axis=1)

        column = ['title', 'text']
        df = pd.DataFrame([[head, body]], columns=column)
        new_data = data.append(df)

        new_data = new_data.iloc[:, :].values
        cv_head = CountVectorizer(max_features=5000)
        mat_head = cv_head.fit_transform(new_data[:, 0]).todense()

        cv_body = CountVectorizer(max_features=5000)
        mat_body = cv_body.fit_transform(new_data[:, 1]).todense()

        X_mat = np.hstack((mat_head, mat_body))
        prediction = loaded_model.predict(X_mat[6000:])

        result = ""
        if prediction[-1] == 0:
            result = 'Fake'
        elif prediction[-1] == 1:
            result = 'Real'
        print(prediction[-1])

        return render_template('detect.html', result=result)

    return render_template('detect.html')


app.run(debug=True)
