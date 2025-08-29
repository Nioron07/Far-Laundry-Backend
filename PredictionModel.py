import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pytz
import sqlalchemy
import logging
import pickle
import os
from functools import lru_cache

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

# Model persistence functions
def save_model(model: RandomForestRegressor, model_name: str):
    """Save trained model to disk"""
    try:
        os.makedirs('models', exist_ok=True)
        with open(f'models/{model_name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model {model_name} saved successfully")
    except Exception as e:
        logger.exception(f"Failed to save model {model_name}")

def load_model(model_name: str) -> RandomForestRegressor:
    """Load trained model from disk"""
    try:
        with open(f'models/{model_name}_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model {model_name} loaded successfully")
        return model
    except FileNotFoundError:
        raise Exception(f"Model {model_name} not found")

def CreateModel(machineType: str, db: sqlalchemy.engine.base.Engine):
    """Create and train model with all available data"""
    # Use all training data with optimized query
    query = """
        SELECT hall, month, weekday, hour, minute, year, day, washers_available, dryers_available
        FROM laundry
        ORDER BY date_added DESC
    """
    
    try:
        df = pd.read_sql(query, con=db)
        logger.info(f"Loaded {len(df)} records for training")
    except Exception as e:
        logger.exception("Failed to load training data")
        raise
    
    if df.empty:
        raise Exception("No training data available")
    
    # Prepare features and target
    feature_columns = ["hall", "month", "weekday", "hour", "minute", "year", "day"]
    X = df[feature_columns]
    y = df[f"{machineType}_available"]
    
    # Train-test split with fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Keep your original hyperparameters
    rf = RandomForestRegressor(
        n_estimators=300,  # Your original value
        max_depth=10,      # Your original value
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Log training score for monitoring
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    logger.info(f"{machineType} model - Train score: {train_score:.3f}, Test score: {test_score:.3f}")
    
    return rf

@lru_cache(maxsize=32)
def get_recent_data_cached(hall: str, day: int, month: int, year: int, current_hour: int, db_str: str):
    """Cached version of recent data retrieval to avoid repeated DB calls"""
    # Note: We pass db as string identifier since we can't cache with actual DB object
    # In practice, you'd implement this differently based on your caching strategy
    pass

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
        current_hour = -1
    
    predictions_dict = {}
    
    if is_today and current_hour >= 0:
        # Optimized query with better indexing strategy
        stmt = sqlalchemy.text("""
            SELECT washers_available, dryers_available, hour
            FROM (
                SELECT 
                    washers_available, dryers_available, hour,
                    ROW_NUMBER() OVER (PARTITION BY hour ORDER BY date_added DESC) AS rn
                FROM laundry
                WHERE hall = :hall
                AND day = :day
                AND month = :month
                AND year = :year
                AND hour <= :current_hour
            ) ranked
            WHERE rn = 1
            ORDER BY hour
        """)
        
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
        except Exception as e:
            logger.exception("Error fetching recent data from the database.")
            recent_data = []
        
        # Process measured data
        for row in recent_data:
            predictions_dict[row[2]] = int(row[machineNum])
        
        # Predict remaining hours
        remaining_hours = [h for h in range(current_hour + 1, 24) if h not in predictions_dict]
        if remaining_hours:
            predict_data = pd.DataFrame({
                "hall": [hall] * len(remaining_hours),
                "month": [day.month] * len(remaining_hours),
                "weekday": [day.weekday()] * len(remaining_hours),
                "hour": remaining_hours,
                "minute": [0] * len(remaining_hours),
                "year": [day.year] * len(remaining_hours),
                "day": [day.day] * len(remaining_hours)
            })
            
            try:
                predictions = model.predict(predict_data)
                for hour, pred in zip(remaining_hours, predictions):
                    predictions_dict[hour] = max(0, int(round(pred)))  # Ensure non-negative
            except Exception as e:
                logger.exception("Error during prediction.")
                for hour in remaining_hours:
                    predictions_dict[hour] = 0
    else:
        # Predict all 24 hours - vectorized approach
        hours = list(range(24))
        predict_data = pd.DataFrame({
            "hall": [hall] * 24,
            "month": [day.month] * 24,
            "weekday": [day.weekday()] * 24,
            "hour": hours,
            "minute": [0] * 24,
            "year": [day.year] * 24,
            "day": [day.day] * 24
        })
        
        try:
            predictions = model.predict(predict_data)
            predictions_dict = {hour: max(0, int(round(pred))) for hour, pred in zip(hours, predictions)}
        except Exception as e:
            logger.exception("Error during prediction.")
            predictions_dict = {hour: 0 for hour in hours}
    
    # Calculate min/max efficiently
    if predictions_dict:
        values = list(predictions_dict.values())
        min_val = min(values)
        max_val = max(values)
        
        min_hour = next(hour for hour, val in predictions_dict.items() if val == min_val)
        max_hour = next(hour for hour, val in predictions_dict.items() if val == max_val)
        
        low_index_str = format_hour(min_hour)
        high_index_str = format_hour(max_hour)
    else:
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
    
    # Vectorized prediction for all 24 hours
    df_24 = pd.DataFrame({
        "hall": [row["hall"]] * 24,
        "month": [row["month"]] * 24,
        "weekday": [row["weekday"]] * 24,
        "hour": np.arange(24),
        "minute": [row["minute"]] * 24,
        "year": [row["year"]] * 24,
        "day": [row["day"]] * 24,
    })
    
    washer_preds = washers.predict(df_24)
    dryer_preds = dryers.predict(df_24)
    
    # Combined availability score
    combined_preds = (washer_preds + dryer_preds) / 2.0
    best_hour = int(np.argmax(combined_preds))
    
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
            "month": [iterDate.month],
            "weekday": [iterDate.weekday()],
            "hour": [0],
            "minute": [0],
            "year": [iterDate.year],
            "day": [iterDate.day],
        })
        
        best_hour_string = GetOptimumTimeDay(washers, dryers, df)
        
        timeArr.append({
            "time": iterDate,
            "bestTime": best_hour_string
        })
        
        iterDate += datetime.timedelta(days=step)
    
    return timeArr

def GetWholeWeekPrediction(model: RandomForestRegressor, hall: str, db: sqlalchemy.engine.base.Engine, machineNum: int):
    start_day = datetime.datetime.now(tz)
    combined_predictions = {}
    full_labels = []
    now = datetime.datetime.now(tz)
    
    for i in range(7):
        current_day = start_day + datetime.timedelta(days=i)
        day_of_week = current_day.weekday()
        day_name = dayOfWeekDict.get(day_of_week, "Unknown")
        
        is_today = (
            current_day.year == now.year and
            current_day.month == now.month and
            current_day.day == now.day
        )
        
        if is_today:
            current_hour = now.hour
            
            # Optimized query for today's data
            stmt = sqlalchemy.text("""
                SELECT washers_available, dryers_available, hour
                FROM (
                    SELECT 
                        washers_available, dryers_available, hour,
                        ROW_NUMBER() OVER (PARTITION BY hour ORDER BY date_added DESC) AS rn
                    FROM laundry
                    WHERE hall = :hall
                    AND day = :day
                    AND month = :month
                    AND year = :year
                    AND hour <= :current_hour
                ) ranked
                WHERE rn = 1
                ORDER BY hour
            """)
            
            try:
                with db.connect() as conn:
                    recent_data = conn.execute(
                        stmt,
                        parameters={
                            "hall": hall,
                            "day": current_day.day,
                            "month": current_day.month,
                            "year": current_day.year,
                            "current_hour": current_hour
                        }
                    ).fetchall()
            except Exception as e:
                logger.exception(f"Error fetching recent data for {day_name}")
                recent_data = []
            
            # Process measured hours
            measured_hours = set()
            for row in recent_data:
                hour, value = row[2], int(row[machineNum])
                label = f"{day_name} {format_hour(hour)}"
                combined_predictions[label] = value
                full_labels.append(label)
                measured_hours.add(hour)
            
            # Predict remaining hours
            remaining_hours = [h for h in range(current_hour + 1, 24)]
            if remaining_hours:
                predict_data = pd.DataFrame({
                    "hall": [hall] * len(remaining_hours),
                    "month": [current_day.month] * len(remaining_hours),
                    "weekday": [day_of_week] * len(remaining_hours),
                    "hour": remaining_hours,
                    "minute": [0] * len(remaining_hours),
                    "year": [current_day.year] * len(remaining_hours),
                    "day": [current_day.day] * len(remaining_hours)
                })
                
                try:
                    predictions = model.predict(predict_data)
                    for hour, pred in zip(remaining_hours, predictions):
                        label = f"{day_name} {format_hour(hour)}"
                        combined_predictions[label] = max(0, int(round(pred)))
                        full_labels.append(label)
                except Exception as e:
                    logger.exception(f"Error during prediction for {day_name}")
                    for hour in remaining_hours:
                        label = f"{day_name} {format_hour(hour)}"
                        combined_predictions[label] = 0
                        full_labels.append(label)
        else:
            # Predict all 24 hours for future days
            hours = list(range(24))
            predict_data = pd.DataFrame({
                "hall": [hall] * 24,
                "month": [current_day.month] * 24,
                "weekday": [day_of_week] * 24,
                "hour": hours,
                "minute": [0] * 24,
                "year": [current_day.year] * 24,
                "day": [current_day.day] * 24
            })
            
            try:
                predictions = model.predict(predict_data)
                for hour, pred in zip(hours, predictions):
                    label = f"{day_name} {format_hour(hour)}"
                    combined_predictions[label] = max(0, int(round(pred)))
                    full_labels.append(label)
            except Exception as e:
                logger.exception(f"Error during prediction for {day_name}")
                for hour in hours:
                    label = f"{day_name} {format_hour(hour)}"
                    combined_predictions[label] = 0
                    full_labels.append(label)
    
    # Calculate min/max efficiently
    if combined_predictions:
        values = list(combined_predictions.values())
        min_val = min(values)
        max_val = max(values)
        
        min_label = next(label for label in full_labels if combined_predictions[label] == min_val)
        max_label = next(label for label in full_labels if combined_predictions[label] == max_val)
    else:
        min_label = max_label = "N/A"
    
    # Prepare final result
    predictions = {label: combined_predictions[label] for label in full_labels}
    predictions["Low"] = min_label
    predictions["High"] = max_label
    
    return predictions

@lru_cache(maxsize=24)
def format_hour(hour):
    """Cached hour formatting function"""
    hour_int = int(hour)  
    period = "AM" if hour_int < 12 else "PM"
    hour_formatted = 12 if hour_int % 12 == 0 else hour_int % 12  
    return f"{hour_formatted}:00{period}"

def getLabel():
    now_central = datetime.datetime.now(tz)
    weekday_str = dayOfWeekDict[now_central.weekday()]
    hour_str = format_hour(now_central.hour)
    return f"{weekday_str} {hour_str}"