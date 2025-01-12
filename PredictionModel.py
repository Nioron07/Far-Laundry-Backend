import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pytz

dayOfWeekDict = {
    6: "Sunday",
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday"
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
    dtr = RandomForestRegressor(max_depth=10, n_estimators=200)
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
    # Create an array of all hours
    hours = list(range(24))
    
    # Create a DataFrame with 24 rows (one per hour)
    data = {
        "What Hall?": [hall] * 24,
        "Month": [day.month] * 24,
        "Weekday": [day.weekday()] * 24,
        "Hour": hours
    }
    df = pd.DataFrame(data)

    # Predict all 24 hours in a single call
    raw_predictions = model.predict(df)
    # Round predictions to integers
    preds_rounded = np.rint(raw_predictions).astype(int)

    # Get minimum and maximum predictions and their indices
    min_val = preds_rounded.min()
    max_val = preds_rounded.max()
    min_idx = preds_rounded.argmin()
    max_idx = preds_rounded.argmax()

    # Convert the predictions into a dict: {hour -> prediction}
    predictions_dict = dict(zip(hours, preds_rounded))

    # You mentioned a helper like "format_hour(i)"—assuming it returns a string:
    low_index_str = format_hour(hours[min_idx])
    high_index_str = format_hour(hours[max_idx])

    return {
        "Predictions": predictions_dict,
        "Low": low_index_str,
        "High": high_index_str
    }

def GetOptimumTime(washers: RandomForestRegressor, dryers: RandomForestRegressor, hall: str, startDay: datetime, endDay: datetime, step: int):
    timeArr = []
    df = pd.DataFrame(columns=["What Hall?", "Month", "Weekday", "Hour"])
    iterDate: datetime = startDay
    while (iterDate <= endDay):
        df['What Hall?'] = [hall]
        df['Month'] = [iterDate.month]
        df['Weekday'] = [iterDate.weekday()]
        timeArr.append({'time': iterDate,"bestTime": GetOptimumTimeDay(washers, dryers, df)})
        iterDate += datetime.timedelta(days=step)
    return timeArr

def GetOptimumTimeDay(washers, dryers, df):
    bestTime = 0
    for i in range (24):
        df['Hour'] = [i]
        avgPredInt = (washers.predict(df)[0] + dryers.predict(df)[0])/2
        if (avgPredInt > bestTime):
                bestTime = avgPredInt
                bestTimeIndex = f"{format_hour(i)}"
    return bestTimeIndex

def GetWholeWeekPrediction(model: RandomForestRegressor, hall: str):
    predictions = {}
    day = datetime.datetime.now(tz)
    low = 100
    lowIndex = ""
    highIndex = ""
    high = -100
    for i in range (7):
        newDay = (day + datetime.timedelta(days=i))
        df = pd.DataFrame(columns=["What Hall?", "Month", "Weekday", "Hour"])
        df['What Hall?'] = [hall]
        df['Month'] = [newDay.month]
        df['Weekday'] = [newDay.weekday()]
        for x in range (24):
            df['Hour'] = [x]
            predInt = round(model.predict(df)[0])
            if (predInt > high):
                high = predInt
                highIndex = f"{dayOfWeekDict[newDay.weekday()]} {format_hour(x)}"
            if(predInt < low):
                low = predInt
                lowIndex = f"{dayOfWeekDict[newDay.weekday()]} {format_hour(x)}"
            predictions[f"{dayOfWeekDict[newDay.weekday()]} {format_hour(x)}"] = predInt
    predictions["High"] = highIndex
    predictions['Low'] = lowIndex
    return predictions

def format_hour(hour):
    hour_int = int(hour)  # Ensure hour is an integer
    period = "AM" if hour_int < 12 else "PM"
    hour_formatted = 12 if hour_int % 12 == 0 else hour_int % 12  # Convert to 12-hour format
    return f"{hour_formatted}:00{period}"

def getLabel():
    now_central = datetime.datetime.now(tz)
    weekday_str = dayOfWeekDict[now_central.weekday()]
    hour_str = format_hour(now_central.hour)
    return f"{weekday_str} {hour_str}"