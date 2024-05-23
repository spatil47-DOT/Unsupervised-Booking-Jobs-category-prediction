"""
Author- Suraj Prakash Patil
Date- 16/05/2024
API calls, prediction function call for Unsupervised Learning
Sample format for Postman API call-
{"Job Description":  ["Booking.com has variety of jobs. About the sales jobs.."]}
"""


'''
5)	Deploying the model-
a.	As a rest api endpoint
b.	Mobil app model
'''

from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import prediction

## Initializing the flask app
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201

            return {"error":"Invalid format."}

        except Exception as error:
            return {'error': error}

class GetPredictionOutput(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            print("data")
            print(data)
            predict = prediction.predict_mpg(data)
            predictOutput = predict
            print("predictOutput")
            print(predictOutput)
            return {'predict':predictOutput}

        except Exception as error:
            return {'error': error}

api.add_resource(Test,'/')
api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)