# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request, make_response 
import flask_cors
import PredictionModel
import SQLConnect
from datetime import datetime, time as dt_time
import threading
import pytz
import logging
import sqlalchemy
import schedule
import time as time_module

logger = logging.getLogger()

# timezone
tz = pytz.timezone("US/Central")
washerModel = None
dryerModel = None
db = None

# Global lock for thread safety during model updates
model_lock = threading.Lock()

def initialize_db():
    """Initialize database connection lazily"""
    global db
    if db is None:
        db = SQLConnect.connect_with_connector()
    return db

def load_models():
    """Load pre-trained models from disk if available, otherwise train new ones"""
    global washerModel, dryerModel
    
    try:
        # Try to load pre-trained models
        washerModel = PredictionModel.load_model("washers")
        dryerModel = PredictionModel.load_model("dryers")
        print("Loaded pre-trained models successfully")
        logger.info("Pre-trained models loaded successfully")
    except Exception as load_error:
        print(f"Failed to load pre-trained models: {load_error}")
        logger.warning(f"Failed to load pre-trained models: {load_error}")
        print("Training new models...")
        logger.info("Starting model training from scratch")
        
        try:
            # Test database connection first
            db_conn = initialize_db()
            if db_conn is None:
                raise Exception("Database connection is None")
            
            # Test basic query
            with db_conn.connect() as conn:
                test_result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM laundry LIMIT 1")).fetchone()
                record_count = test_result[0] if test_result else 0
                print(f"Found {record_count} records in database for training")
                logger.info(f"Database check passed: {record_count} records available")
                
                if record_count == 0:
                    raise Exception("No training data available in database")
            
            train_models()
            
        except Exception as train_error:
            print(f"CRITICAL: Failed to train new models: {train_error}")
            logger.exception("CRITICAL: Model training failed completely")
            # Set models to None to indicate failure
            washerModel = None
            dryerModel = None
            raise Exception(f"Model initialization completely failed: {train_error}")

def train_models():
    """Train new models and save them"""
    global washerModel, dryerModel
    
    with model_lock:
        db_conn = initialize_db()
        print("Starting model training process...")
        logger.info("Beginning model training with database connection")
        
        try:
            # Train washer model
            print("Training washer model...")
            washerModel = PredictionModel.CreateModel("washers", db_conn)
            print("Washer model training completed")
            logger.info("Washer model trained successfully")
            
            # Train dryer model  
            print("Training dryer model...")
            dryerModel = PredictionModel.CreateModel("dryers", db_conn)
            print("Dryer model training completed")
            logger.info("Dryer model trained successfully")
            
            # Save models to disk
            print("Saving models to disk...")
            PredictionModel.save_model(washerModel, "washers")
            PredictionModel.save_model(dryerModel, "dryers")
            print("Models trained and saved successfully")
            logger.info("All models trained and saved successfully")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            logger.exception("Detailed model training failure")
            # Re-raise to be caught by load_models
            raise



def should_retrain():
    """Check if it's time to retrain (between midnight and 2 AM)"""
    now = datetime.now(tz)
    return 0 <= now.hour <= 2

def schedule_retraining():
    """Schedule daily retraining at 1 AM"""
    def retrain_job():
        if should_retrain():
            print("Starting scheduled retraining...")
            train_models()
        else:
            print("Skipping retraining - not in low usage hours")
    
    # Schedule for 1 AM every day
    schedule.every().day.at("01:00").do(retrain_job)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

def async_model_init():
    global washerModel, dryerModel  # <-- Move global declaration to the top
    
    try:
        print("=== STARTING MODEL INITIALIZATION ===")
        logger.info("Starting async model initialization")
        
        load_models()
        schedule_retraining()
        
        print("=== MODEL INITIALIZATION COMPLETED SUCCESSFULLY ===")
        logger.info("Model initialization completed successfully")
        
        # Verify models are actually loaded
        if washerModel is None or dryerModel is None:
            raise Exception("Models are None after initialization")
            
        print(f"Final check - Washer model: {type(washerModel)}, Dryer model: {type(dryerModel)}")
        
    except Exception as e:
        print(f"=== MODEL INITIALIZATION FAILED: {e} ===")
        logger.exception("CRITICAL: Failed to initialize models")
        
        # Set global state to indicate failure
        washerModel = None
        dryerModel = None
# Start model initialization in background
init_thread = threading.Thread(target=async_model_init, daemon=True)
init_thread.start()

# Wait a moment for models to potentially load (for Cloud Run cold starts)
import time
time.sleep(2)

# creating a Flask app 
app = Flask(__name__)
app.json.sort_keys = False
cors = flask_cors.CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET']) 
def home():
    return jsonify({'data': 'hello world', 'models_ready': washerModel is not None and dryerModel is not None}) 

@app.route('/current/<int:hall>', methods=['GET']) 
def current(hall):
    db_conn = initialize_db()
    
    # Optimized query with index hint
    stmt = sqlalchemy.text("""
        SELECT washers_available, dryers_available, date_added 
        FROM laundry
        WHERE hall = :hall
        ORDER BY date_added DESC
        LIMIT 1
    """)
    
    try:
        with db_conn.connect() as conn:
            result = conn.execute(stmt, parameters={"hall": hall}).fetchone()
            
        if not result:
            return jsonify({'error': 'No data found for hall'}), 404
            
        return jsonify({
            'Washing Machines': result[0],
            "Dryers": result[1],
            "Timestamp": result[2].strftime('%-I:%M%p').upper()
        })
    except Exception as e:
        logger.exception(e)
        return jsonify({'error': 'Database error'}), 500

@app.route('/currentTime', methods=['GET']) 
def getTime():
    return jsonify({'Time': PredictionModel.getLabel()}) 
  
@app.route('/today/<int:hall>', methods=['GET']) 
def today(hall):
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    db_conn = initialize_db()
    
    with model_lock:
        return jsonify({
            'Washing Machines': PredictionModel.GetWholeDayPrediction(washerModel, hall, datetime.now(tz), db_conn, 0),
            "Dryers": PredictionModel.GetWholeDayPrediction(dryerModel, hall, datetime.now(tz), db_conn, 1)
        })

@app.route('/day/<int:hall>/dayOfMonth/<int:dayOfMonth>/month/<int:month>', methods=['GET']) 
def day(hall, dayOfMonth, month):
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    db_conn = initialize_db()
    
    try:
        target_date = datetime(datetime.now(tz).year, month, dayOfMonth)
    except ValueError:
        return jsonify({'error': 'Invalid date'}), 400
    
    with model_lock:
        return jsonify({
            'Washing Machines': PredictionModel.GetWholeDayPrediction(washerModel, hall, target_date, db_conn, 0),
            "Dryers": PredictionModel.GetWholeDayPrediction(dryerModel, hall, target_date, db_conn, 1)
        }) 
  
@app.route('/week/<int:hall>', methods=['GET'])
def week(hall):
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    db_conn = initialize_db()
    
    with model_lock:
        return jsonify({
            'Current Time': PredictionModel.getLabel(), 
            'Washing Machines': PredictionModel.GetWholeWeekPrediction(washerModel, hall, db_conn, 0),
            "Dryers": PredictionModel.GetWholeWeekPrediction(dryerModel, hall, db_conn, 1)
        })

@app.route('/optimumTime/<int:hall>/startDay/<int:startDay>/endDay/<int:endDay>/step/<int:step>', methods=['GET']) 
def optimumTime(hall, startDay, endDay, step):
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    try:
        start_date = datetime.fromtimestamp(startDay / 1000.0)
        end_date = datetime.fromtimestamp(endDay / 1000.0)
    except (ValueError, OSError):
        return jsonify({'error': 'Invalid timestamp'}), 400
    
    with model_lock:
        return jsonify({
            "Optimum Time": PredictionModel.GetOptimumTime(washerModel, dryerModel, hall, start_date, end_date, step)
        })

# New endpoint for detailed day statistics
@app.route('/dayStats/<int:hall>/dayOfMonth/<int:dayOfMonth>/month/<int:month>', methods=['GET']) 
def dayStats(hall, dayOfMonth, month):
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    db_conn = initialize_db()
    
    try:
        target_date = datetime(datetime.now(tz).year, month, dayOfMonth)
    except ValueError:
        return jsonify({'error': 'Invalid date'}), 400
    
    with model_lock:
        washer_data = PredictionModel.GetWholeDayPrediction(washerModel, hall, target_date, db_conn, 0)
        dryer_data = PredictionModel.GetWholeDayPrediction(dryerModel, hall, target_date, db_conn, 1)
        
        return jsonify({
            'date': target_date.strftime('%Y-%m-%d'),
            'hall': hall,
            'washers': {
                'predictions': washer_data['Predictions'],
                'statistics': washer_data['Statistics'],
                'high': washer_data['High'],
                'low': washer_data['Low']
            },
            'dryers': {
                'predictions': dryer_data['Predictions'],
                'statistics': dryer_data['Statistics'],
                'high': dryer_data['High'],
                'low': dryer_data['Low']
            }
        })

# New endpoint for detailed week statistics
@app.route('/weekStats/<int:hall>', methods=['GET'])
def weekStats(hall):
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    db_conn = initialize_db()
    
    with model_lock:
        washer_data = PredictionModel.GetWholeWeekPrediction(washerModel, hall, db_conn, 0)
        dryer_data = PredictionModel.GetWholeWeekPrediction(dryerModel, hall, db_conn, 1)
        
        return jsonify({
            'hall': hall,
            'current_time': PredictionModel.getLabel(),
            'washers': {
                'predictions': {k: v for k, v in washer_data.items() if k not in ['Low', 'High', 'Statistics']},
                'statistics': washer_data.get('Statistics', {}),
                'high': washer_data.get('High', 'N/A'),
                'low': washer_data.get('Low', 'N/A')
            },
            'dryers': {
                'predictions': {k: v for k, v in dryer_data.items() if k not in ['Low', 'High', 'Statistics']},
                'statistics': dryer_data.get('Statistics', {}),
                'high': dryer_data.get('High', 'N/A'),
                'low': dryer_data.get('Low', 'N/A')
            }
        })

# New endpoint for real-time analytics
@app.route('/analytics/<int:hall>', methods=['GET'])
def analytics(hall):
    """Provides real-time analytics and trends"""
    if not (washerModel and dryerModel):
        return jsonify({'error': 'Models not ready'}), 503
        
    db_conn = initialize_db()
    
    try:
        # Get current week data
        with model_lock:
            washer_week = PredictionModel.GetWholeWeekPrediction(washerModel, hall, db_conn, 0)
            dryer_week = PredictionModel.GetWholeWeekPrediction(dryerModel, hall, db_conn, 1)
        
        # Calculate trends and patterns
        current_time = datetime.now(tz)
        current_day = current_time.strftime('%A')[:3]  # Get short day name
        current_hour = current_time.hour
        
        # Find best times for today
        today_washer_data = {k: v for k, v in washer_week.items() 
                            if k.startswith(current_day) and k not in ['Low', 'High', 'Statistics']}
        today_dryer_data = {k: v for k, v in dryer_week.items() 
                           if k.startswith(current_day) and k not in ['Low', 'High', 'Statistics']}
        
        # Calculate combined availability for recommendations
        combined_scores = {}
        for washer_key in today_washer_data:
            if washer_key in today_dryer_data:
                combined_scores[washer_key] = today_washer_data[washer_key] + today_dryer_data[washer_key]
        
        # Sort by combined availability
        best_times = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            'hall': hall,
            'current_time': current_time.strftime('%I:%M %p'),
            'current_day': current_day,
            'recommendations': {
                'best_times_today': [{'time': time, 'score': score} for time, score in best_times],
                'current_availability': {
                    'washers': today_washer_data.get(f"{current_day} {current_time.strftime('%I:%M%p')}", 0),
                    'dryers': today_dryer_data.get(f"{current_day} {current_time.strftime('%I:%M%p')}", 0)
                },
                'trends': {
                    'washer_weekly_avg': washer_week.get('Statistics', {}).get('mean', 0),
                    'dryer_weekly_avg': dryer_week.get('Statistics', {}).get('mean', 0),
                    'peak_availability_time': washer_week.get('High', 'N/A'),
                    'low_availability_time': washer_week.get('Low', 'N/A')
                }
            }
        })
    except Exception as e:
        logger.exception(e)
        return jsonify({'error': 'Analytics calculation failed'}), 500

@app.route('/contribute', methods=['POST']) 
def contribute():
    db_conn = initialize_db()
    
    try:
        data = request.json.get('data')
        if not data or len(data) < 3:
            return make_response("Invalid data format", 400)
            
        # Optimized insert with prepared statement
        stmt = sqlalchemy.text("""
            INSERT INTO laundry (washers_available, dryers_available, hall, month, weekday, hour, minute, year, date_added, day) 
            VALUES (:washers, :dryers, :hall, :month, :weekday, :hour, :minute, :year, :date_added, :day)
        """)
        
        now = datetime.now(tz=tz)
        with db_conn.connect() as conn:
            conn.execute(stmt, parameters={
                "washers": data[0], 
                "dryers": data[1], 
                "hall": data[2], 
                "month": now.month, 
                "weekday": now.weekday(), 
                "hour": now.hour, 
                "minute": now.minute, 
                "year": now.year, 
                "date_added": now.strftime("%Y-%m-%d %H:%M:%S"), 
                "day": now.day
            })
            conn.commit()
            
        return make_response("POST request succeeded", 200)
    except Exception as e:
        logger.exception(e)
        return make_response("POST request failed", 500)

@app.route('/health', methods=['GET'])
def health():
    """Comprehensive health check endpoint"""
    db_conn = initialize_db()
    
    # Check database connectivity
    db_healthy = False
    try:
        with db_conn.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1")).fetchone()
        db_healthy = True
    except Exception as e:
        logger.exception("Database health check failed")
    
    # Check model status
    models_ready = washerModel is not None and dryerModel is not None
    
    status = 'healthy' if (db_healthy and models_ready) else 'degraded'
    if not db_healthy and not models_ready:
        status = 'critical'
    
    response_data = {
        'status': status,
        'models_ready': models_ready,
        'washer_model_loaded': washerModel is not None,
        'dryer_model_loaded': dryerModel is not None,
        'db_connected': db_healthy,
        'prediction_intervals': '10 minutes',
        'features': ['10-min intervals', 'statistics', 'analytics', 'trends'],
        'timestamp': datetime.now(tz).isoformat()
    }
    
    # Return appropriate status code
    if status == 'critical':
        return jsonify(response_data), 503
    elif status == 'degraded':
        return jsonify(response_data), 202  # Accepted but not complete
    else:
        return jsonify(response_data), 200

@app.route('/model-status', methods=['GET'])
def model_status():
    """Detailed model status for debugging"""
    return jsonify({
        'washer_model': {
            'loaded': washerModel is not None,
            'type': type(washerModel).__name__ if washerModel else None,
            'features': washerModel.n_features_in_ if washerModel else None
        },
        'dryer_model': {
            'loaded': dryerModel is not None,
            'type': type(dryerModel).__name__ if dryerModel else None,
            'features': dryerModel.n_features_in_ if dryerModel else None
        },
        'database': {
            'connected': db is not None
        },
        'initialization': {
            'thread_alive': init_thread.is_alive() if 'init_thread' in globals() else False
        }
    })

@app.route('/startup-check', methods=['GET'])
def startup_check():
    """Quick endpoint to check if app is starting up properly"""
    try:
        # Check if we can import our modules
        import PredictionModel
        import SQLConnect
        
        # Check basic database connection
        db_test = False
        try:
            test_db = initialize_db()
            db_test = test_db is not None
        except Exception as e:
            print(f"Database connection test failed: {e}")
        
        return jsonify({
            'app_started': True,
            'imports_working': True,
            'database_connection': db_test,
            'models': {
                'washer_loaded': washerModel is not None,
                'dryer_loaded': dryerModel is not None
            },
            'init_thread_alive': init_thread.is_alive() if 'init_thread' in globals() else False,
            'timestamp': datetime.now(tz).isoformat()
        })
    except Exception as e:
        return jsonify({
            'app_started': True,
            'error': str(e),
            'imports_working': False,
            'timestamp': datetime.now(tz).isoformat()
        }), 500
# driver function 
if __name__ == '__main__': 
    app.run(debug=True)