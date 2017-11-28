from flask import Flask
from flask import render_template, jsonify, request
import pandas as pd
import os, json
from featureExtraction import extractFeatures
from predict import predictProbabilityForDifferentFeatures

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
APP_DATA=os.path.join(APP_ROOT,'data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    ques1 = request.args.get('ques1')
    ques2 = request.args.get('ques2')
    extractFeatures(ques1,ques2)

    df = pd.read_csv(os.path.join(APP_DATA, 'quora_features_test.csv'))
    predictionJSON=predictProbabilityForDifferentFeatures()

    feature_list = ['fuzz_qratio', 'fuzz_WRatio', 'wmd', 'norm_wmd', 'cosine_distance',
                    'jaccard_distance', 'euclidean_distance', 'braycurtis_distance', 'cosSim']

    response_JSON = {}

    for feature in feature_list:
        if df.iloc[0][feature] < 1.0:
            response_JSON[feature] = str(df.iloc[0][feature]*100)
        else:
            response_JSON[feature] = str(df.iloc[0][feature])

    outputJSON=[predictionJSON,response_JSON]

    return jsonify(outputJSON)


if __name__ == '__main__':
    app.run(debug=True)
