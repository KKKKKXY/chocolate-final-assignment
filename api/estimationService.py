# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from getResources import GetResources
from collections import OrderedDict
import pandas as pd
import numpy as np


app = Flask(__name__)
app.model = GetResources.getModel()
app.lookupDict = GetResources.getLookupDict()


@app.route('/estimate', methods=['GET'])
def estimate():
    argList = request.args.to_dict(flat=False)
    queryDF=pd.DataFrame.from_dict(OrderedDict(argList))
    try:
        for feat in queryDF.columns:
            if feat in app.lookupDict:
                try:
                    queryDF[feat] = app.lookupDict[feat].transform(np.ravel(queryDF[feat]))
                except:
                    queryDF[feat] = app.lookupDict[feat].transform(np.ravel(['Unknown']))
            else:
                queryDF[feat] = queryDF[feat].astype("float64")
                
                
    except:
        return "Error - check params"
    estimatedRating = app.model.predict(queryDF)[0]
    
    
    return jsonify(rating=str(estimatedRating))


@app.route('/', methods=['GET'])
def hello():
    return 'hello'


if __name__ == '__main__':
    app.run(debug=True)
