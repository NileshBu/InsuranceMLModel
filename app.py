#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:59:41 2021

@author: nilesh
"""
from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import pickle


def preProcess(data):
    
    #Education
    data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
    #Job
    data['Job'] = data['Job'].fillna(data['Job'].mode()[0])
    #Communication
    data['Communication'] = data['Communication'].fillna('Missing')
    #Outcome
    data['Outcome'] = data['Outcome'].fillna('Mising')
    #Marital
    data['Marital'] = data['Marital'].fillna(data['Marital'].mode()[0])


    for i in ['CallStart', 'CallEnd']:
        data[i] = pd.to_datetime(data[i])
    data['CallDur'] = ((data['CallEnd']-data['CallStart']).dt.seconds)/60

    data['CallHour'] = data['CallStart'].dt.hour
    data = data.drop(['CallStart', 'CallEnd'], axis = 1)
    return data
    
    
app= Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():
    dfs = []
    data = request.get_json()
    dfs.append(pd.DataFrame([data]))
    df = pd.concat(dfs, ignore_index=True, sort=False)
    try:
        preprocessed_df=preProcess(df)
    except:
        return jsonify("Error occured while preprocessing your data for our model!")
    try:
        predictions= model.predict(preprocessed_df)
    except:
        return jsonify("Error occured while processing your data into our model!")
    print("done")
    return make_response(jsonify(np.array2string(predictions)),200)    



if __name__=='__main__':
    modelfile = 'final_model.sav'
    model = pickle.load(open(modelfile, 'rb'))
    app.run(host='0.0.0.0', debug=True)
