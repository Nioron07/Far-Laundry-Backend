# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request, make_response 
import flask_cors
import PredictionModel
import SQLConnect
from datetime import datetime
import threading
import pytz
import logging
import sqlalchemy
logger = logging.getLogger()
# timezone
tz = pytz.timezone("US/Central")
washerModel = None
dryerModel = None
db = SQLConnect.connect_with_connector()
def Retrain():
    global washerModel
    global dryerModel
    washerModel = PredictionModel.CreateModel("washers", db)
    dryerModel = PredictionModel.CreateModel("dryers", db)
    print("Retrained")
    threading.Timer(86400.0, Retrain).start()
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
    stmt = sqlalchemy.text(
        """SELECT washers_available, dryers_available, date_added FROM laundry
            WHERE hall = :hall
            ORDER BY date_added DESC
            LIMIT 1"""
    )
    try:
        # Using a with statement ensures that the connection is always released
        # back into the pool at the end of statement (even if an error occurs)
        with db.connect() as conn:
            recent_data = conn.execute(stmt, parameters={"hall": hall}).fetchall()
            print(recent_data)
    except Exception as e:
        # If something goes wrong, handle the error in this section. This might
        # involve retrying or adjusting parameters depending on the situation.
        # [START_EXCLUDE]
        logger.exception(e)
    return jsonify({'Washing Machines': recent_data[0][0],
                    "Dryers": recent_data[0][1],
                    "Timestamp": recent_data[0][2].strftime('%-I:%M%p').upper()}) 

@app.route('/currentTime', methods = ['GET']) 
def getTime():
    return jsonify({'Time': PredictionModel.getLabel()}) 
  
@app.route('/today/<int:hall>', methods = ['GET']) 
def today(hall):
    return jsonify({'Washing Machines': PredictionModel.GetWholeDayPrediction(washerModel, hall, datetime.now(tz), db, 0),
                    "Dryers": PredictionModel.GetWholeDayPrediction(dryerModel, hall, datetime.now(tz), db, 1)}) 

@app.route('/day/<int:hall>/dayOfMonth/<int:dayOfMonth>/month/<int:month>', methods = ['GET']) 
def day(hall, dayOfMonth, month):
    return jsonify({'Washing Machines': PredictionModel.GetWholeDayPrediction(washerModel, hall, datetime(datetime.now(tz).year, month, dayOfMonth), db, 0),
                    "Dryers": PredictionModel.GetWholeDayPrediction(dryerModel, hall, datetime(datetime.now(tz).year, month, dayOfMonth), db, 1)}) 
  
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
        stmt = sqlalchemy.text(
            "INSERT INTO laundry (washers_available, dryers_available, hall, month, weekday, hour, minute, year, date_added, day) VALUES (:washers, :dryers, :hall, :month, :weekday, :hour, :minute, :year, :date_added, :day)"
        )
        try:
            # Using a with statement ensures that the connection is always released
            # back into the pool at the end of statement (even if an error occurs)
            now = datetime.now(tz=tz)
            with db.connect() as conn:
                conn.execute(stmt, parameters={"washers": data[0], "dryers": data[1], "hall": data[2], "month": now.month, "weekday": now.weekday(), "hour": now.hour, "minute": now.minute, "year": now.year, "date_added": now.strftime("%Y-%m-%d %H:%M:%S"), "day": now.day})
                conn.commit()
                print("commited changes")
        except Exception as e:
            # If something goes wrong, handle the error in this section. This might
            # involve retrying or adjusting parameters depending on the situation.
            # [START_EXCLUDE]
            logger.exception(e)

        return make_response("POST request succeded", 200)
    except:
        return make_response("POST request failed", 201)
  

# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 