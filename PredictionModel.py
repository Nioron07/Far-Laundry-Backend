import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
import pytz
import sqlalchemy
import logging
logger = logging.getLogger()

dayOfWeekDict = {
    6: "Sun",
    0: "Mon",
    1: "Tues",
    2: "Wed",
    3: "Thurs",
    4: "Fri",
    5: "Sat"
}

tz = pytz.timezone("US/Central")
def CreateModel(machineType: str, db: sqlalchemy.engine.base.Engine):
    query = "SELECT * FROM laundry;"
    df = pd.read_sql(query, con=db)
    print(df)
    #Assigns Independent and Dependent variables in the form of X and y
    X = df.drop(["washers_available", "dryers_available", "date_added", "id"], axis = 1)
    y = df[f"{machineType}_available"]

    #Creates testing data sets and training data sets with training making up 80% of the origninal dataset and testing being 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Defines and Fits each of the respective classifiers
    # dtr = RandomForestRegressor(max_depth=10, n_estimators=200, n_jobs=-1)
    # dtr.fit(X_train, y_train)
    dtr = SGDRegressor()
    dtr.partial_fit(X_train, y_train)
    return dtr

def GetWholeDayPrediction(model: SGDRegressor, hall: str, day: datetime, db: sqlalchemy.engine.base.Engine, machineNum: int):
    hours = list(range(24))
    
    data = {
        "hall": [hall] * 24,
        "month": [day.month] * 24,
        "weekday": [day.weekday()] * 24,
        "hour": hours,
        "minute": [0] * 24,
        "year": [day.year] * 24,
        "day": [day.day] * 24
    }
    df = pd.DataFrame(data)

    raw_predictions = model.predict(df)
    preds_rounded = [int(round(x)) for x in raw_predictions]
    
    predictions_dict = {int(h): int(preds_rounded[i]) for i, h in enumerate(hours)}

    min_val = min(preds_rounded)
    max_val = max(preds_rounded)
    min_idx = preds_rounded.index(min_val)
    max_idx = preds_rounded.index(max_val)
    
    low_index_str = format_hour(hours[min_idx])
    high_index_str = format_hour(hours[max_idx])

    if day.day == datetime.datetime.now(tz).day and day.month == datetime.datetime.now(tz).month and day.year == datetime.datetime.now(tz).year:
        stmt = sqlalchemy.text(
        """SELECT washers_available, dryers_available, date_added FROM laundry
            WHERE hall = :hall
            ORDER BY date_added DESC
            LIMIT 1"""
            # WITH cte AS (
            #     SELECT 
            #         t.*,
            #         ROW_NUMBER() OVER (PARTITION BY t.hour ORDER BY t.id) AS rn
            #     FROM laundry t
            #     WHERE t.hall = 0
            #     AND t.day = 25
            #     AND t.month = 1
            #     AND t.year = 2025
            #     AND t.hour <= 20  -- or your upper limit
            # )
            # SELECT *
            # FROM cte
            # WHERE rn = 1
            # ORDER BY hour;
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
        print(datetime.datetime.now(tz).hour)
        print(recent_data)
        predictions_dict[f"{datetime.datetime.now(tz).hour}"] = recent_data[0][machineNum]
    return {
        "Predictions": predictions_dict,
        "Low": low_index_str,
        "High": high_index_str
    }

def GetOptimumTimeDay(washers: SGDRegressor, 
                      dryers: SGDRegressor, 
                      df: pd.DataFrame) -> str:
    row = df.iloc[0]
    
    df_24 = pd.DataFrame({
        "hall":  [row["hall"]]*24,
        "month":       [row["month"]]*24,
        "weekday":     [row["weekday"]]*24,
        "hour":        np.arange(24),
        "minute": [row["minute"]]*24,
        "year": [row["year"]]*24,
        "day": [row["day"]]*24,
    })
    
    washer_preds = washers.predict(df_24)
    dryer_preds  = dryers.predict(df_24)
    
    avg_preds = (washer_preds + dryer_preds) / 2.0
    
    best_hour = int(np.argmax(avg_preds))
    
    return format_hour(best_hour)

def GetOptimumTime(washers: SGDRegressor, 
                   dryers: SGDRegressor, 
                   hall: str, 
                   startDay: datetime.datetime, 
                   endDay: datetime.datetime, 
                   step: int):
    timeArr = []
    iterDate = startDay
    while iterDate <= endDay:
        df = pd.DataFrame({
            "hall": [hall],
            "month":      [iterDate.month],
            "weekday":    [iterDate.weekday()],
            "hour":       [0],
            "minute": [0],
            "year": [iterDate.year],
            "day": [iterDate.day],
        })
        
        best_hour_string = GetOptimumTimeDay(washers, dryers, df)
        
        timeArr.append({
            "time":     iterDate,
            "bestTime": best_hour_string
        })
        
        iterDate += datetime.timedelta(days=step)
    
    return timeArr

def GetWholeWeekPrediction(model: SGDRegressor, hall: str, db: sqlalchemy.engine.base.Engine, machineNum: int):
    start_day = datetime.datetime.now(tz)

    data = {
            "hall": [],
            "month":      [],
            "weekday":    [],
            "hour":       [],
            "minute": [],
            "year": [],
            "day": [],
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
            data["minute"].append(0)
            data["year"].append(current_day.year)
            data["day"].append(current_day.day)

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
    print(datetime.datetime.now(tz).hour)
    print(recent_data)
    predictions[f"{dayOfWeekDict[datetime.datetime.now(tz).weekday()]} {format_hour(datetime.datetime.now(tz).hour)}"] = recent_data[0][machineNum]
    predictions["Low"] = full_labels[min_idx]
    predictions["High"] = full_labels[max_idx]

    return predictions


def format_hour(hour):
    hour_int = int(hour)  
    period = "AM" if hour_int < 12 else "PM"
    hour_formatted = 12 if hour_int % 12 == 0 else hour_int % 12  
    return f"{hour_formatted}:00{period}"

def getLabel():
    now_central = datetime.datetime.now(tz)
    weekday_str = dayOfWeekDict[now_central.weekday()]
    hour_str = format_hour(now_central.hour)
    return f"{weekday_str} {hour_str}"