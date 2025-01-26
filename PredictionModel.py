import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    #Assigns Independent and Dependent variables in the form of X and y
    X = df.drop(["washers_available", "dryers_available", "date_added", "id"], axis = 1)
    y = df[f"{machineType}_available"]

    #Creates testing data sets and training data sets with training making up 80% of the origninal dataset and testing being 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Defines and Fits each of the respective classifiers
    dtr = RandomForestRegressor(max_depth=10, n_estimators=300, n_jobs=-1)
    dtr.fit(X_train, y_train)
    return dtr

def GetWholeDayPrediction(
    model: RandomForestRegressor,
    hall: str,
    day: datetime.datetime,
    db: sqlalchemy.engine.base.Engine,
    machineNum: int
):
    now = datetime.datetime.now(tz)
    is_today = (
        day.year == now.year and
        day.month == now.month and
        day.day == now.day
    )
    
    if is_today:
        current_hour = now.hour
    else:
        current_hour = -1  # Indicates that all hours need to be predicted
    
    if is_today and current_hour >= 0:
        # Query the database for measured data up to the current hour
        stmt = sqlalchemy.text(
        """WITH cte AS (
                SELECT 
                    t.*,
                    ROW_NUMBER() OVER (PARTITION BY t.hour ORDER BY t.id DESC) AS rn
                FROM laundry t
                WHERE t.hall = :hall
                AND t.day = :day
                AND t.month = :month
                AND t.year = :year
                AND t.hour <= :current_hour
            )
            SELECT washers_available, dryers_available, hour
            FROM cte
            WHERE rn = 1
            ORDER BY date_added;"""
        )
        
        try:
            with db.connect() as conn:
                recent_data = conn.execute(
                    stmt,
                    parameters={
                        "hall": hall,
                        "day": day.day,
                        "month": day.month,
                        "year": day.year,
                        "current_hour": current_hour
                    }
                ).fetchall()
            print(recent_data)
        except Exception as e:
            logger.exception("Error fetching recent data from the database.")
            recent_data = []
        
        # Extract measured hours and their values
        measured_hours = [row[2] for row in recent_data]
        measured_values = [int(row[machineNum]) for row in recent_data]
        
        # Prepare predictions for remaining hours
        remaining_hours = list(range(current_hour + 1, 24))
        if remaining_hours:
            data = {
                "hall": [hall] * len(remaining_hours),
                "month": [day.month] * len(remaining_hours),
                "weekday": [day.weekday()] * len(remaining_hours),
                "hour": remaining_hours,
                "minute": [0] * len(remaining_hours),
                "year": [day.year] * len(remaining_hours),
                "day": [day.day] * len(remaining_hours)
            }
            df_predict = pd.DataFrame(data)
            
            try:
                raw_predictions = model.predict(df_predict)
                preds_rounded = [int(round(x)) for x in raw_predictions]
            except Exception as e:
                logger.exception("Error during prediction.")
                preds_rounded = [0] * len(remaining_hours)  # Fallback to zeros or handle appropriately
            
            predicted_dict = {hour: pred for hour, pred in zip(remaining_hours, preds_rounded)}
        else:
            predicted_dict = {}
        
        # Combine measured and predicted data
        predictions_dict = {hour: value for hour, value in zip(measured_hours, measured_values)}
        predictions_dict.update(predicted_dict)
    else:
        # Predict for all 24 hours
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
        
        try:
            raw_predictions = model.predict(df)
            preds_rounded = [int(round(x)) for x in raw_predictions]
        except Exception as e:
            logger.exception("Error during prediction.")
            preds_rounded = [0] * 24  # Fallback to zeros or handle appropriately
        
        predictions_dict = {hour: pred for hour, pred in zip(hours, preds_rounded)}
    
    if predictions_dict:
        # Determine minimum and maximum predictions
        min_val = min(predictions_dict.values())
        max_val = max(predictions_dict.values())
        
        # Find the first occurrence of min and max
        min_hours = [hour for hour, val in predictions_dict.items() if val == min_val]
        max_hours = [hour for hour, val in predictions_dict.items() if val == max_val]
        
        # Format the first min and max hours
        low_index_str = format_hour(min_hours[0]) if min_hours else "N/A"
        high_index_str = format_hour(max_hours[0]) if max_hours else "N/A"
    else:
        min_val = max_val = 0
        low_index_str = high_index_str = "N/A"
    
    return {
        "Predictions": predictions_dict,
        "Low": low_index_str,
        "High": high_index_str
    }

def GetOptimumTimeDay(washers: RandomForestRegressor, 
                      dryers: RandomForestRegressor, 
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

def GetOptimumTime(washers: RandomForestRegressor, 
                   dryers: RandomForestRegressor, 
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

def GetWholeWeekPrediction(model: RandomForestRegressor, hall: str, db: sqlalchemy.engine.base.Engine, machineNum: int):
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

        for hour in range(datetime.datetime.now(tz).hour if i == 0 else 0, 24 - datetime.datetime.now(tz).hour if i == 0 else 24):
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