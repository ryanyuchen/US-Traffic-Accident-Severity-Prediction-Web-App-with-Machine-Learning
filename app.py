import csv
import json
import ast
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, make_response
import pandas as pd

app = Flask(__name__)
# setting up an application whose name is app with Flask


@app.route('/')
@app.route('/index.html')
# both connect to explore page
def index():
    #ml_algorithms = ['Random Forest', 'XGBoost']
    return render_template('index.html')


@app.route('/explore.html')
def Analysis():
    return render_template('explore.html')

@app.route('/dataset.html')
def Dataset():
    return render_template('dataset.html')


@app.route('/about.html')
def About():
    return render_template('about.html')

@app.route('/severitybycounty1_YC.html')
def Severitybycounty():
    return render_template('severitybycounty1_YC.html')

@app.route('/severitybyyear_YC.html')
def Severitybyyear():
    return render_template('severitybyyear_YC.html')

@app.route('/featureviz_YC.html')
def features():
    return render_template('featureviz_YC.html')

import numpy as np
import pickle
# model = pickle.load(open('models/RF.pickle', 'rb'))


@app.route("/model_picker" , methods=['GET', 'POST'])
def model_picker():

    selected_model = None
    if request.method == "POST":
        selected_model = request.json['model_selection']
        print(selected_model)
    
    return redirect(url_for('predict', picked_model = selected_model)) #going back to explore page
    # selected_model = request.form.get('model_selection')
    # return(str(selected_model)) # just to see what select is


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model = pickle.load(open('models/15wRF_limit.pickle', 'rb'))
    # print(model)
    # selected_model = request.args.get('picked_model')
    # print(selected_model)
    # if selected_model=="XGBoost": 
    #     model = pickle.load(open('models/8w_XGV_serApp.pickle', 'rb'))
    # else:
    #     print("you are selecting a wrong model!!!")

    input_string = [x for x in request.form.values()]

    selected_model = str(input_string[0])
    if selected_model=="xgboost":
        print("selected model is " + selected_model)
        model = pickle.load(open('models/15wXGB_limit.pickle', 'rb'))
        features_values_string_numeric = input_string.copy()[1:]

        features_values_string_numeric[2] = 0.15
        features_values_string_numeric[3] = 0.65

        features_values = [float(x) for x in features_values_string_numeric]
        p = []
        p.append(features_values[5])
        p.append(features_values[4])
        p.append(features_values[0])
        p.append(features_values[7])
        p.append(features_values[8])
        p.append(features_values[1])
        p.append(features_values[6])
        p.append(features_values[9])
        p.append(features_values[3])
        p.append(features_values[2])

        input_string_predictors = 'Your input values: Distance, ' + str(features_values[0]) + ' miles; ' + 'Duration, ' +  str(features_values[1]) + ' minutes; '\
            + 'City: ' + str(input_string[3]) + '; Street: ' +  str(input_string[4]) + '; Longitude of the start point: ' + str(features_values[4])\
            + '; Latitude of the start points: ' + str(features_values[5]) + '; Air Pression: ' + str(features_values[6]) + ' inches; '\
            + 'Temperature: ' + str(features_values[7]) + ' F; '\
            + 'Humidity: ' + str(features_values[8]) + '%; '\
            + 'Wind Speed: ' + str(features_values[9]) + ' mph'

        final_features = [np.array(p)]
        prediction = model.predict(final_features)
        output = prediction[0]

        return render_template('index.html', picked_model='The model you picked is ' + selected_model,
                input_values=input_string_predictors, prediction_text='The predicted accident severity is {}'.format(output))

    
    features_values_string_numeric = input_string.copy()[1:]

    features_values_string_numeric[2] = 0.15
    features_values_string_numeric[3] = 0.65

    features_values = [float(x) for x in features_values_string_numeric]

    input_string_predictors = 'Your input values: Distance, ' + str(features_values[0]) + ' miles; ' + 'Duration, ' +  str(features_values[1]) + ' minutes; '\
        + 'City: ' + str(input_string[3]) + '; Street: ' +  str(input_string[4]) + '; Longitude of the start point: ' + str(features_values[4])\
        + '; Latitude of the start points: ' + str(features_values[5]) + '; Air Pression: ' + str(features_values[6]) + ' inches; '\
        + 'Temperature: ' + str(features_values[7]) + ' F; '\
        + 'Humidity: ' + str(features_values[8]) + '%; '\
        + 'Wind Speed: ' + str(features_values[9]) + ' mph'

    final_features = [np.array(features_values)]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', picked_model='The model you picked is ' + selected_model,
            input_values=input_string_predictors, prediction_text='The predicted accident severity is {}'.format(output))

if __name__ == '__main__':
    app.run(host='localhost', port=6242, debug=True)
    # app.run()