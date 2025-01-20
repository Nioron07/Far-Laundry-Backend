import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pytz

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
def CreateModel(machineType: str):
    df = pd.concat([pd.read_csv("./Data Files/GoogleFormData.csv"), pd.read_csv("./Data Files/WebAppData.csv")], axis=0)

    #Assigns Independent and Dependent variables in the form of X and y
    X = df.drop(["How many Washing Machines are Available?", "How many Dryers are Available?"], axis = 1)
    y = df["How many " + machineType + " are Available?"]

    #Creates testing data sets and training data sets with training making up 80% of the origninal dataset and testing being 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Defines and Fits each of the respective classifiers
    dtr = RandomForestRegressor(max_depth=10, n_estimators=200, n_jobs=-1)
    dtr.fit(X_train, y_train)
    return dtr

def GetCurrentPrediction(model: RandomForestRegressor, hall: str):
    day = datetime.datetime.now(tz)
    df = pd.DataFrame(columns=["What Hall?", "Month", "Weekday", "Hour"])
    df['What Hall?'] = [hall]
    df['Month'] = [day.month]
    df['Weekday'] = [day.weekday()]
    df['Hour'] = [day.hour]
    return round(model.predict(df)[0])

def GetWholeDayPrediction(model: RandomForestRegressor, hall: str, day: datetime):
    hours = list(range(24))
    
    data = {
        "What Hall?": [hall] * 24,
        "Month": [day.month] * 24,
        "Weekday": [day.weekday()] * 24,
        "Hour": hours
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
        "What Hall?":  [row["What Hall?"]]*24,
        "Month":       [row["Month"]]*24,
        "Weekday":     [row["Weekday"]]*24,
        "Hour":        np.arange(24)
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
            "What Hall?": [hall],
            "Month":      [iterDate.month],
            "Weekday":    [iterDate.weekday()],
            "Hour":       [0]
        })
        
        best_hour_string = GetOptimumTimeDay(washers, dryers, df)
        
        timeArr.append({
            "time":     iterDate,
            "bestTime": best_hour_string
        })
        
        iterDate += datetime.timedelta(days=step)
    
    return timeArr

def GetWholeWeekPrediction(model: RandomForestRegressor, hall: str):
    start_day = datetime.datetime.now(tz)

    data = {
        "What Hall?": [],
        "Month": [],
        "Weekday": [],
        "Hour": []
    }

    full_labels = []

    for i in range(7):
        current_day = start_day + datetime.timedelta(days=i)
        day_of_week = current_day.weekday()  
        day_name = dayOfWeekDict[day_of_week] 

        for hour in range(24):
            data["What Hall?"].append(hall)
            data["Month"].append(current_day.month)
            data["Weekday"].append(day_of_week)
            data["Hour"].append(hour)

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