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
from scipy import stats

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

# Model persistence functions (unchanged)
def save_model(model: RandomForestRegressor, model_name: str):
    try:
        os.makedirs('models', exist_ok=True)
        with open(f'models/{model_name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model {model_name} saved successfully")
    except Exception as e:
        logger.exception(f"Failed to save model {model_name}")

def load_model(model_name: str) -> RandomForestRegressor:
    try:
        with open(f'models/{model_name}_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model {model_name} loaded successfully")
        return model
    except FileNotFoundError:
        raise Exception(f"Model {model_name} not found")

def CreateModel(machineType: str, db: sqlalchemy.engine.base.Engine):
    """Create and train model with all available data"""
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

def calculate_statistics(predictions):
    """Calculate comprehensive statistics for predictions"""
    if not predictions:
        return {}
    
    values = list(predictions.values())
    
    # Basic statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Percentiles
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    
    # Availability periods (consecutive periods with machines available)
    availability_periods = []
    current_period = 0
    
    for val in values:
        if val > 0:
            current_period += 1
        else:
            if current_period > 0:
                availability_periods.append(current_period)
            current_period = 0
    
    if current_period > 0:
        availability_periods.append(current_period)
    
    # Peak and low periods
    peak_threshold = mean_val + std_val
    low_threshold = max(0, mean_val - std_val)
    
    peak_periods = [k for k, v in predictions.items() if v >= peak_threshold]
    low_periods = [k for k, v in predictions.items() if v <= low_threshold]
    
    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "min": int(min_val),
        "max": int(max_val),
        "q25": round(q25, 2),
        "q75": round(q75, 2),
        "availability_percentage": round(len([v for v in values if v > 0]) / len(values) * 100, 1),
        "peak_periods": peak_periods[:5],  # Top 5 peak periods
        "low_periods": low_periods[:5],    # Top 5 low periods
        "avg_availability_duration": round(np.mean(availability_periods) if availability_periods else 0, 1),
        "longest_availability_period": max(availability_periods) if availability_periods else 0,
        "total_availability_periods": len(availability_periods)
    }

def GetWholeDayPrediction(
    model: RandomForestRegressor,
    hall: str,
    day: datetime.datetime,
    db: sqlalchemy.engine.base.Engine,
    machineNum: int
):
    """Generate predictions every 10 minutes for a full day"""
    now = datetime.datetime.now(tz)
    is_today = (
        day.year == now.year and
        day.month == now.month and
        day.day == now.day
    )
    
    predictions_dict = {}
    
    # Generate 10-minute intervals for 24 hours (144 intervals)
    intervals = []
    for hour in range(24):
        for minute in [0, 10, 20, 30, 40, 50]:
            intervals.append((hour, minute))
    
    if is_today:
        current_hour = now.hour
        current_minute = (now.minute // 10) * 10  # Round to nearest 10 minutes
        current_interval_index = current_hour * 6 + (current_minute // 10)
        
        # Get recent measured data
        stmt = sqlalchemy.text("""
            SELECT washers_available, dryers_available, hour, minute
            FROM (
                SELECT 
                    washers_available, dryers_available, hour, minute,
                    ROW_NUMBER() OVER (PARTITION BY hour, FLOOR(minute/10)*10 ORDER BY date_added DESC) AS rn
                FROM laundry
                WHERE hall = :hall
                AND day = :day
                AND month = :month
                AND year = :year
                AND (hour < :current_hour OR (hour = :current_hour AND minute <= :current_minute))
            ) ranked
            WHERE rn = 1
            ORDER BY hour, minute
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
                        "current_hour": current_hour,
                        "current_minute": current_minute
                    }
                ).fetchall()
        except Exception as e:
            logger.exception("Error fetching recent data from the database.")
            recent_data = []
        
        # Process measured data
        for row in recent_data:
            hour, minute = row[2], (row[3] // 10) * 10  # Round minute to 10-minute interval
            key = f"{hour:02d}:{minute:02d}"
            predictions_dict[key] = int(row[machineNum])
        
        # Predict remaining intervals
        remaining_intervals = intervals[current_interval_index + 1:]
        if remaining_intervals:
            predict_data = []
            keys = []
            
            for hour, minute in remaining_intervals:
                predict_data.append({
                    "hall": hall,
                    "month": day.month,
                    "weekday": day.weekday(),
                    "hour": hour,
                    "minute": minute,
                    "year": day.year,
                    "day": day.day
                })
                keys.append(f"{hour:02d}:{minute:02d}")
            
            df_predict = pd.DataFrame(predict_data)
            
            try:
                predictions = model.predict(df_predict)
                for key, pred in zip(keys, predictions):
                    predictions_dict[key] = max(0, int(round(pred)))
            except Exception as e:
                logger.exception("Error during prediction.")
                for key in keys:
                    predictions_dict[key] = 0
    else:
        # Predict all intervals for non-today dates
        predict_data = []
        keys = []
        
        for hour, minute in intervals:
            predict_data.append({
                "hall": hall,
                "month": day.month,
                "weekday": day.weekday(),
                "hour": hour,
                "minute": minute,
                "year": day.year,
                "day": day.day
            })
            keys.append(f"{hour:02d}:{minute:02d}")
        
        df_predict = pd.DataFrame(predict_data)
        
        try:
            predictions = model.predict(df_predict)
            for key, pred in zip(keys, predictions):
                predictions_dict[key] = max(0, int(round(pred)))
        except Exception as e:
            logger.exception("Error during prediction.")
            predictions_dict = {key: 0 for key in keys}
    
    # Calculate statistics
    stats = calculate_statistics(predictions_dict)
    
    # Find min/max with times
    if predictions_dict:
        values = list(predictions_dict.values())
        min_val = min(values)
        max_val = max(values)
        
        min_time = next(time for time, val in predictions_dict.items() if val == min_val)
        max_time = next(time for time, val in predictions_dict.items() if val == max_val)
        
        # Convert to readable format
        min_hour, min_minute = map(int, min_time.split(':'))
        max_hour, max_minute = map(int, max_time.split(':'))
        
        low_index_str = format_time(min_hour, min_minute)
        high_index_str = format_time(max_hour, max_minute)
    else:
        low_index_str = high_index_str = "N/A"
    
    return {
        "Predictions": predictions_dict,
        "Low": low_index_str,
        "High": high_index_str,
        "Statistics": stats
    }

def GetWholeWeekPrediction(model: RandomForestRegressor, hall: str, db: sqlalchemy.engine.base.Engine, machineNum: int):
    """Generate predictions for a full week with 10-minute intervals"""
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
        
        # Generate 10-minute intervals for this day
        for hour in range(24):
            for minute in [0, 10, 20, 30, 40, 50]:
                time_key = f"{hour:02d}:{minute:02d}"
                label = f"{day_name} {format_time(hour, minute)}"
                
                if is_today:
                    current_hour = now.hour
                    current_minute = (now.minute // 10) * 10
                    
                    if hour < current_hour or (hour == current_hour and minute <= current_minute):
                        # Try to get measured data
                        stmt = sqlalchemy.text("""
                            SELECT washers_available, dryers_available
                            FROM laundry
                            WHERE hall = :hall
                            AND day = :day
                            AND month = :month
                            AND year = :year
                            AND hour = :hour
                            AND ABS(minute - :minute) <= 5
                            ORDER BY ABS(minute - :minute), date_added DESC
                            LIMIT 1
                        """)
                        
                        try:
                            with db.connect() as conn:
                                result = conn.execute(
                                    stmt,
                                    parameters={
                                        "hall": hall,
                                        "day": current_day.day,
                                        "month": current_day.month,
                                        "year": current_day.year,
                                        "hour": hour,
                                        "minute": minute
                                    }
                                ).fetchone()
                            
                            if result:
                                combined_predictions[label] = int(result[machineNum])
                            else:
                                # Predict if no measured data
                                prediction = predict_single_interval(model, hall, current_day, hour, minute)
                                combined_predictions[label] = max(0, int(round(prediction)))
                        except Exception as e:
                            logger.exception(f"Error fetching data for {label}")
                            prediction = predict_single_interval(model, hall, current_day, hour, minute)
                            combined_predictions[label] = max(0, int(round(prediction)))
                    else:
                        # Future intervals - predict
                        prediction = predict_single_interval(model, hall, current_day, hour, minute)
                        combined_predictions[label] = max(0, int(round(prediction)))
                else:
                    # Other days - predict all intervals
                    prediction = predict_single_interval(model, hall, current_day, hour, minute)
                    combined_predictions[label] = max(0, int(round(prediction)))
                
                full_labels.append(label)
    
    # Calculate statistics
    stats = calculate_statistics(combined_predictions)
    
    # Find min/max
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
    predictions["Statistics"] = stats
    
    return predictions

def predict_single_interval(model: RandomForestRegressor, hall: str, day: datetime.datetime, hour: int, minute: int):
    """Helper function to predict a single interval"""
    try:
        df = pd.DataFrame([{
            "hall": hall,
            "month": day.month,
            "weekday": day.weekday(),
            "hour": hour,
            "minute": minute,
            "year": day.year,
            "day": day.day
        }])
        return model.predict(df)[0]
    except Exception as e:
        logger.exception("Error in single interval prediction")
        return 0

@lru_cache(maxsize=144)
def format_time(hour, minute):
    """Cached time formatting function for 10-minute intervals"""
    period = "AM" if hour < 12 else "PM"
    hour_formatted = 12 if hour % 12 == 0 else hour % 12
    return f"{hour_formatted}:{minute:02d}{period}"

def getLabel():
    now_central = datetime.datetime.now(tz)
    weekday_str = dayOfWeekDict[now_central.weekday()]
    # Round to nearest 10 minutes
    minute_rounded = (now_central.minute // 10) * 10
    time_str = format_time(now_central.hour, minute_rounded)
    return f"{weekday_str} {time_str}"

# Keep existing optimum time functions but update for 10-minute intervals
def GetOptimumTimeDay(washers: RandomForestRegressor, 
                      dryers: RandomForestRegressor, 
                      df: pd.DataFrame) -> str:
    row = df.iloc[0]
    
    # Generate predictions for all 10-minute intervals
    intervals_data = []
    for hour in range(24):
        for minute in [0, 10, 20, 30, 40, 50]:
            intervals_data.append({
                "hall": row["hall"],
                "month": row["month"],
                "weekday": row["weekday"],
                "hour": hour,
                "minute": minute,
                "year": row["year"],
                "day": row["day"]
            })
    
    df_intervals = pd.DataFrame(intervals_data)
    
    washer_preds = washers.predict(df_intervals)
    dryer_preds = dryers.predict(df_intervals)
    
    # Combined availability score
    combined_preds = (washer_preds + dryer_preds) / 2.0
    best_idx = int(np.argmax(combined_preds))
    
    best_hour = best_idx // 6
    best_minute = (best_idx % 6) * 10
    
    return format_time(best_hour, best_minute)

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
        
        best_time_string = GetOptimumTimeDay(washers, dryers, df)
        
        timeArr.append({
            "time": iterDate,
            "bestTime": best_time_string
        })
        
        iterDate += datetime.timedelta(days=step)
    
    return timeArr