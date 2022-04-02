import flask
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json
import pandas as pd

app = flask.Flask(__name__)
xgb_with_h = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json(force=True)
    data_json = json.loads(data_json)
    data_dict = np.array(list(data_json.values())).flatten()    

    response = {}
    response = xgb_with_h.predict_proba(data_dict.reshape(1,-1)).tolist()
    return flask.jsonify(response)
    print(type(response()))

if __name__ == "__main__": 
    app.run(debug=True)
