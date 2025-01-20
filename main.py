# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request, make_response 
import flask_cors
import PredictionModel
import DataScraper
from datetime import datetime
import pandas as pd
import threading
import pytz
from google.cloud.sql.connector import Connector
import sqlalchemy
from dotenv import load_dotenv
import os
load_dotenv()
def get_sqlalchemy_engine(

    instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME"),
    db_user = os.getenv("USER"),
    db_pass = os.getenv("PASSWORD"),
    db_name = os.getenv("DB")
) -> sqlalchemy.engine.Engine:
    """
    Creates a SQLAlchemy engine using the Cloud SQL Python Connector.
    """
    connector = Connector()  # instantiate once, globally or in a factory

    def getconn():
        conn = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name
        )
        return conn

    # Create the connection pool via SQLAlchemy
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn
    )
    return engine
# timezone
tz = pytz.timezone("US/Central")
washerModel = None
dryerModel = None
engine = get_sqlalchemy_engine()

def Retrain():
    global washerModel
    global dryerModel
    threading.Timer(86400.0, Retrain).start()
    print("Retrained")
    washerModel = PredictionModel.CreateModel("Washing Machines")
    dryerModel = PredictionModel.CreateModel("Dryers")

def GetData():
    DataScraper.scrape_laundry_summary()
    threading.Timer(300.0, GetData).start()

GetData()
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
    query = """
        SELECT 
            washers_available,
            dryers_available,
            hall,
            month,
            weekday,
            hour,
            minute
        FROM laundry
        WHERE hall = :hall
        ORDER BY id DESC   -- or your preferred method for "latest" row
        LIMIT 1
    """

    with engine.begin() as conn:
        row = conn.execute(sqlalchemy.text(query), {"hall": hall}).fetchone()

    if row is None:
        return jsonify({"error": f"No data found for hall {hall}"}), 404

    washers_available = row[0]
    dryers_available  = row[1]
    # row[2] = hall (already known)
    month   = row[3]
    weekday = row[4]
    hour    = row[5]
    minute  = row[6]

    # Convert to 12-hour format: "4:05pm"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    ampm = "am" if hour < 12 else "pm"

    # Example final: "Wed 4:05pm" or just "4:05pm"
    # Below includes weekday, remove if you only want the time.
    timestamp_str = f"{hour_12}:{minute:02d}{ampm}"

    return jsonify({
        "Washing Machines": washers_available,
        "Dryers": dryers_available,
        "Timestamp": timestamp_str
    })

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
        # 1) Get the 'data' list from the JSON body of the POST request.
        #    e.g. data might look like [5, 2, 0, 9, 3, 14, 0]
        data = request.json['data']
        
        # 2) Check if a row with the exact same values already exists in the DB.
        check_query = """
            SELECT COUNT(*) AS cnt
            FROM laundry
            WHERE washers_available = %s
              AND dryers_available  = %s
              AND hall              = %s
              AND month             = %s
              AND weekday           = %s
              AND hour              = %s
              AND minute            = %s
        """
        
        # 3) If not a duplicate, insert into the table.
        insert_query = """
            INSERT INTO laundry
                (washers_available, dryers_available, hall, month, weekday, hour, minute)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
        """
        
        with engine.begin() as conn:
            # Run the check query
            result = conn.execute(sqlalchemy.text(check_query), tuple(data))
            row_count = result.fetchone()[0]
            
            if row_count > 0:
                # A matching row already exists
                return make_response("POST request contains duplicate data", 202)
            
            # Otherwise, insert the new row
            conn.execute(sqlalchemy.text(insert_query), tuple(data))
            
        return make_response("POST request succeeded", 200)
    
    except Exception as e:
        # Log or print the error if needed
        print("Error in contribute():", e)
        return make_response("POST request failed", 201)
  

# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 