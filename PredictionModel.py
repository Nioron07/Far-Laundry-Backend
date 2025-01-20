import datetime
import numpy as np
import pandas as pd
import pytz

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# If using the Cloud SQL Python Connector:
#   pip install "cloud-sql-python-connector[pymysql]" sqlalchemy
from google.cloud.sql.connector import Connector
import sqlalchemy

import os
from dotenv import load_dotenv
load_dotenv()
# Dictionary for converting weekday() integer to a string label
dayOfWeekDict = {
    6: "Sun",
    0: "Mon",
    1: "Tues",
    2: "Wed",
    3: "Thurs",
    4: "Fri",
    5: "Sat"
}

# Set your timezone
tz = pytz.timezone("US/Central")

# -----------------------------------------------------------------------------
# A) Setup your Cloud SQL connection (via Python Connector)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# B) Pull your data from the "laundry" table
# -----------------------------------------------------------------------------
def get_data_from_mysql(engine: sqlalchemy.engine.Engine) -> pd.DataFrame:
    """
    Fetches all data from a single table named "laundry".
    """
    query = "SELECT * FROM laundry;"
    df_laundry = pd.read_sql(query, con=engine)
    
    # Ensure df_laundry has columns:
    #   washers_available, dryers_available, hall, month, weekday, hour, minute
    # If that's correct, just return it:
    return df_laundry


# -----------------------------------------------------------------------------
# C) Create and train the model
# -----------------------------------------------------------------------------
def CreateModel(machineType: str,
                engine: sqlalchemy.engine.Engine) -> RandomForestRegressor:
    """
    Creates a RandomForestRegressor model based on data pulled from your
    'laundry' table, predicting either washers_available or dryers_available.
    """
    # 1) Load data from MySQL
    df = get_data_from_mysql(engine)

    # 2) Figure out which column we should predict
    #    Let's say if the user says "Washing Machines" we map that to washers_available,
    #    otherwise "Dryers" -> dryers_available. You can handle other logic as needed.
    machineTypeLower = machineType.lower()
    if "wash" in machineTypeLower:
        target_col = "washers_available"
    else:
        target_col = "dryers_available"

    # 3) Define X and y
    #    We drop both washers_available and dryers_available so that neither
    #    is in the features. Then we pick the correct target_col for y.
    X = df.drop(["washers_available", "dryers_available"], axis=1)
    y = df[target_col]

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 5) Define and fit the RandomForestRegressor
    model = RandomForestRegressor(max_depth=10, n_estimators=200, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------------------------
# D) Predictions
# -----------------------------------------------------------------------------
def GetCurrentPrediction(model: RandomForestRegressor, hall: str) -> int:
    """
    Predicts availability (washers or dryers) for the current moment, given a hall name.
    """
    now = datetime.datetime.now(tz)
    df = pd.DataFrame({
        "hall":    [hall],
        "month":   [now.month],
        "weekday": [now.weekday()],
        "hour":    [now.hour],
        # You can choose to use the exact minute, or 0, etc.
        "minute":  [now.minute]
    })
    return int(round(model.predict(df)[0]))


def GetWholeDayPrediction(model: RandomForestRegressor, hall: str, day: datetime.datetime):
    """
    Generates predictions (washers or dryers) for each hour of a given day.
    By default, sets minute=0 for each hour.
    """
    hours = list(range(24))
    data = {
        "hall":    [hall] * 24,
        "month":   [day.month] * 24,
        "weekday": [day.weekday()] * 24,
        "hour":    hours,
        "minute":  [0] * 24  # or set each minute differently if you want
    }
    df = pd.DataFrame(data)

    raw_predictions = model.predict(df)
    preds_rounded = [int(round(x)) for x in raw_predictions]

    # Create a dict hour -> prediction
    predictions_dict = {}
    for i, hr in enumerate(hours):
        predictions_dict[hr] = preds_rounded[i]

    min_val = min(preds_rounded)
    max_val = max(preds_rounded)
    min_idx = preds_rounded.index(min_val)
    max_idx = preds_rounded.index(max_val)

    low_index_str = format_hour(hours[min_idx])
    high_index_str = format_hour(hours[max_idx])

    return {
        "Predictions": predictions_dict,
        "Low": low_index_str,
        "High": high_index_str
    }


def GetOptimumTimeDay(washers: RandomForestRegressor, 
                      dryers: RandomForestRegressor, 
                      df: pd.DataFrame) -> str:
    """
    Given a single row DataFrame describing 'hall', 'month', 'weekday', 'hour', 'minute',
    we generate 24 rows from 0..23 hours (minute=0) and predict washers & dryers availability.
    Then we pick the hour that has the maximum average of washers+dryers available.
    """
    row = df.iloc[0]
    
    df_24 = pd.DataFrame({
        "hall":    [row["hall"]]*24,
        "month":   [row["month"]]*24,
        "weekday": [row["weekday"]]*24,
        "hour":    np.arange(24),
        "minute":  [0]*24  # you could change the minute logic if needed
    })
    
    washer_preds = washers.predict(df_24)
    dryer_preds  = dryers.predict(df_24)

    # Average them to find best hour for combined usage
    avg_preds = (washer_preds + dryer_preds) / 2.0

    best_hour_index = int(np.argmax(avg_preds))

    return format_hour(best_hour_index)


def GetOptimumTime(washers: RandomForestRegressor, 
                   dryers: RandomForestRegressor, 
                   hall: str, 
                   startDay: datetime.datetime, 
                   endDay: datetime.datetime, 
                   step: int):
    """
    Generates the best hour of each day (from startDay to endDay, stepping by 'step' days).
    """
    timeArr = []
    iterDate = startDay
    while iterDate <= endDay:
        df = pd.DataFrame({
            "hall":    [hall],
            "month":   [iterDate.month],
            "weekday": [iterDate.weekday()],
            "hour":    [0],
            "minute":  [0]  # or handle minutes differently
        })
        
        best_hour_string = GetOptimumTimeDay(washers, dryers, df)
        
        timeArr.append({
            "time":     iterDate,
            "bestTime": best_hour_string
        })
        
        iterDate += datetime.timedelta(days=step)
    
    return timeArr


def GetWholeWeekPrediction(model: RandomForestRegressor, hall: str):
    """
    Generates predictions for the next 7 days (for every hour).
    We'll keep 'minute' = 0 for each hour.
    """
    start_day = datetime.datetime.now(tz)

    data = {
        "hall":    [],
        "month":   [],
        "weekday": [],
        "hour":    [],
        "minute":  []
    }

    full_labels = []

    for i in range(7):
        current_day = start_day + datetime.timedelta(days=i)
        day_of_week = current_day.weekday()
        day_name = dayOfWeekDict[day_of_week]

        for hour in range(24):
            data["hall"].append(hall)
            data["month"].append(current_day.month)
            data["weekday"].append(day_of_week)
            data["hour"].append(hour)
            data["minute"].append(0)  # or something else

            full_labels.append(f"{day_name} {format_hour(hour)}")

    df = pd.DataFrame(data)

    raw_predictions = model.predict(df)
    preds_rounded = [int(round(x)) for x in raw_predictions]

    min_val = min(preds_rounded)
    max_val = max(preds_rounded)
    min_idx = preds_rounded.index(min_val)
    max_idx = preds_rounded.index(max_val)

    predictions = {}
    for i, label in enumerate(full_labels):
        predictions[label] = preds_rounded[i]

    predictions["Low"] = full_labels[min_idx]
    predictions["High"] = full_labels[max_idx]

    return predictions


# -----------------------------------------------------------------------------
# E) Utility functions
# -----------------------------------------------------------------------------
def format_hour(hour: int) -> str:
    """
    Convert an integer hour (0..23) into a string like "1:00AM" or "12:00PM".
    """
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:00{period}"


def getLabel() -> str:
    """
    Returns a label of the form "Tue 2:00PM" for the *current* time in US/Central.
    """
    now_central = datetime.datetime.now(tz)
    weekday_str = dayOfWeekDict[now_central.weekday()]
    hour_str = format_hour(now_central.hour)
    return f"{weekday_str} {hour_str}"