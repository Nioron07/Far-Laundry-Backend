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
    
    # Create a DataFrame with 24 rows (one row per hour)
    data = {
        "What Hall?": [hall] * 24,
        "Month": [day.month] * 24,
        "Weekday": [day.weekday()] * 24,
        "Hour": hours
    }
    df = pd.DataFrame(data)

    # Predict all 24 hours in a single call
    raw_predictions = model.predict(df)
    # Round predictions to the nearest integer
    # Here we convert them directly to a Python list of Python ints
    preds_rounded = [int(round(x)) for x in raw_predictions]
    
    # Convert the predictions into a dict of Python ints: {hour -> prediction}
    predictions_dict = {int(h): int(preds_rounded[i]) for i, h in enumerate(hours)}

    # Find min and max predictions and their corresponding hours (indices)
    min_val = min(preds_rounded)
    max_val = max(preds_rounded)
    min_idx = preds_rounded.index(min_val)
    max_idx = preds_rounded.index(max_val)
    
    # Convert the hour to a formatted string (assuming this function exists)
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
    # Start from "today" in the given timezone
    start_day = datetime.datetime.now(tz)

    # Prepare containers for building a single DataFrame
    data = {
        "What Hall?": [],
        "Month": [],
        "Weekday": [],
        "Hour": []
    }

    # We'll keep a parallel list of labels (for dictionary keys and min/max)
    full_labels = []

    # Build 7 days × 24 hours = 168 rows
    for i in range(7):
        current_day = start_day + datetime.timedelta(days=i)
        day_of_week = current_day.weekday()  # 0=Monday, 6=Sunday
        day_name = dayOfWeekDict[day_of_week]  # e.g., 'Mon', 'Tue', ...

        for hour in range(24):
            data["What Hall?"].append(hall)
            data["Month"].append(current_day.month)
            data["Weekday"].append(day_of_week)
            data["Hour"].append(hour)

            # For labeling each prediction in the final dict
            full_labels.append(f"{day_name} {format_hour(hour)}")

    # Create the DataFrame of all rows at once
    df = pd.DataFrame(data)

    # Single prediction call (returns a NumPy array)
    raw_predictions = model.predict(df)

    # Convert each prediction to a Python int so it's JSON-serializable
    preds_rounded = [int(round(x)) for x in raw_predictions]

    # Find min and max predictions and their corresponding indices
    min_val = min(preds_rounded)
    max_val = max(preds_rounded)
    min_idx = preds_rounded.index(min_val)
    max_idx = preds_rounded.index(max_val)

    # Build final dictionary of predictions
    predictions = {}
    for i, label in enumerate(full_labels):
        predictions[label] = preds_rounded[i]

    # Attach the "High" and "Low" keys to show which day/hour is min or max
    predictions["Low"] = full_labels[min_idx]
    predictions["High"] = full_labels[max_idx]

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