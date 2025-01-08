# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request, make_response 
import flask_cors
import DataCleaning
import PredictionModel
from datetime import datetime
import pandas as pd
import threading
import pytz

DataCleaning.UpdateDataFiles()

tz = pytz.timezone("US/Central")
washerModel = None
dryerModel = None
def Retrain():
    global washerModel
    global dryerModel
    threading.Timer(86400.0, Retrain).start()
    print("Retrained")
    washerModel = PredictionModel.CreateModel("Washing Machines")
    dryerModel = PredictionModel.CreateModel("Dryers")

Retrain()
# creating a Flask app 
app = Flask(__name__)
app.json.sort_keys = False
cors = flask_cors.CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods = ['GET']) 
def home():
    data = "hello world"
    return jsonify({'data': data}) 

@app.route('/current/<int:hall>', methods = ['GET']) 
def current(hall):
    return jsonify({'Washing Machines': int(PredictionModel.GetCurrentPrediction(washerModel, hall)),
                    "Dryers": int(PredictionModel.GetCurrentPrediction(dryerModel, hall))}) 

@app.route('/currentTime', methods = ['GET']) 
def getTime():
    return jsonify({'Time': PredictionModel.getLabel()}) 
  
@app.route('/today/<int:hall>', methods = ['GET']) 
def today(hall):
    return jsonify({'Washing Machines': PredictionModel.GetWholeDayPrediction(washerModel, hall, datetime.now(tz)),
                    "Dryers": PredictionModel.GetWholeDayPrediction(dryerModel, hall, datetime.now(tz))}) 

@app.route('/day/<int:hall>/dayOfMonth/<int:dayOfMonth>/month/<int:month>', methods = ['GET']) 
def day(hall, dayOfMonth, month):
    return jsonify({'Washing Machines': PredictionModel.GetWholeDayPrediction(washerModel, hall, datetime(datetime.now(tz).year, month, dayOfMonth)),
                    "Dryers": PredictionModel.GetWholeDayPrediction(dryerModel, hall, datetime(datetime.now(tz).year, month, dayOfMonth))}) 
  
@app.route('/week/<int:hall>', methods = ['GET'])
def week(hall):
    return jsonify({'Current Time': PredictionModel.getLabel(), 'Washing Machines': PredictionModel.GetWholeWeekPrediction(washerModel, hall),
                    "Dryers": PredictionModel.GetWholeWeekPrediction(dryerModel, hall)})
@app.route('/optimumTime/<int:hall>/startDay/<int:startDay>/endDay/<int:endDay>/step/<int:step>', methods = ['GET']) 
def optimumTime(hall, startDay, endDay, step):
    return jsonify({"Optimum Time": PredictionModel.GetOptimumTime(washerModel, dryerModel, hall, datetime.fromtimestamp(startDay / 1000.0), datetime.fromtimestamp(endDay / 1000.0), step)})
@app.route('/contribute', methods = ['POST']) 
def contribute():
    try:
        data = request.json['data']
        df = pd.read_csv("Laundry Data Pro\\laundry-API-backend\\Data Files\\WebAppData.csv")
        if (((df['How many Washing Machines are Available?'] == data[0]) & (df['How many Dryers are Available?'] == data[1]) & (df['What Hall?'] == data[2]) & (df['Month'] == data[3]) & (df['Weekday'] == data[4]) & (df['Hour'] == data[5])).any()):
            return make_response("POST request contains duplicate data", 202)
        df.loc[df["What Hall?"].size] = request.json['data']
        df.to_csv("Laundry Data Pro\\laundry-API-backend\\Data Files\\WebAppData.csv", index=False)
        return make_response("POST request succeded", 200)
    except:
        return make_response("POST request failed", 201)
  

# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 