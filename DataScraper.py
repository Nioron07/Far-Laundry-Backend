from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from datetime import datetime
import pytz
from selenium.webdriver.chrome.options import Options
import os
from dotenv import load_dotenv
load_dotenv()

# ============ SETUP FOR HEADLESS CHROME ============
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

tz = pytz.timezone("US/Central")

from google.cloud.sql.connector import Connector
import sqlalchemy

# 1) Instantiate the Connector object
connector = Connector()

def get_sqlalchemy_engine(

    instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME"),
    db_user = os.getenv("USER"),
    db_pass = os.getenv("PASSWORD"),
    db_name = os.getenv("DB")
    ) -> sqlalchemy.engine.Engine:
    """
    Creates a SQLAlchemy engine using the Cloud SQL Python Connector.
    """
    def getconn():
        conn = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name
        )
        return conn

    # 3) Use 'creator=getconn' to create a connection pool via SQLAlchemy
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn
    )
    return engine

# Create a single global engine (or do it inside a main() if you prefer)
engine = get_sqlalchemy_engine()

def scrape_laundry_summary():
    # URLs for your washers/dryers pages
    Og_url = "https://mycscgo.com/laundry/summary/b94db20b-3bf8-4517-9cae-da46d7dd73f6/2303113-025"
    Tr_url = "https://mycscgo.com/laundry/summary/b94db20b-3bf8-4517-9cae-da46d7dd73f6/2303113-026"
    
    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # ---------------------- OG URL SCRAPE ----------------------
        driver.get(Og_url)
        time.sleep(5)  # Wait for the JS to load (adjust as needed)
        
        washers_xpath = '//*[@id="root"]/div/main/div[2]/div/div/div[2]/div[1]/p'
        dryers_xpath  = '//*[@id="root"]/div/main/div[2]/div/div/div[4]/div[1]/p'
        
        try:
            Og_washers = driver.find_element(By.XPATH, washers_xpath).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find Og washers: {e}")
            Og_washers = "0"
        try:
            Og_dryers = driver.find_element(By.XPATH, dryers_xpath).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find Og dryers: {e}")
            Og_dryers = "0"

        # ---------------------- TR URL SCRAPE ----------------------
        driver.get(Tr_url)
        time.sleep(5)  # Wait for the JS to load (adjust as needed)
        
        try:
            Tr_washers = driver.find_element(By.XPATH, washers_xpath).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find Tr washers: {e}")
            Tr_washers = "0"
        try:
            Tr_dryers = driver.find_element(By.XPATH, dryers_xpath).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find Tr dryers: {e}")
            Tr_dryers = "0"
    
    finally:
        # Close the browser
        driver.quit()

    # ---------------------- INSERT INTO DB ----------------------
    # We have two rows of data to insert:
    #  1) [Og_washers, Og_dryers, hall=0, now.month, now.weekday(), now.hour, now.minute]
    #  2) [Tr_washers, Tr_dryers, hall=1, now.month, now.weekday(), now.hour, now.minute]
    now = datetime.now(tz=tz)

    row_og = (
        int(Og_washers),
        int(Og_dryers),
        0,                # hall=0
        now.month,
        now.weekday(),
        now.hour,
        now.minute
    )
    row_tr = (
        int(Tr_washers),
        int(Tr_dryers),
        1,                # hall=1
        now.month,
        now.weekday(),
        now.hour,
        now.minute
    )

    # Insert query (adjust column names if different in your DB)
    insert_sql = """
        INSERT INTO laundry
            (washers_available, dryers_available, hall, month, weekday, hour, minute)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
    """
    
    # Execute the insert in a transaction
    with engine.begin() as conn:
        conn.execute(insert_sql, row_og)
        conn.execute(insert_sql, row_tr)