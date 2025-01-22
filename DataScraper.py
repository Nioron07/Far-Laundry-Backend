from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from datetime import datetime
import pytz
import logging
from selenium.webdriver.chrome.options import Options
import sqlalchemy
logger = logging.getLogger()
    
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
    
# Initialize the WebDriver (Chrome in this example)
# Make sure you've installed ChromeDriver and have it in your PATH.
driver = webdriver.Chrome(options=chrome_options)

tz = pytz.timezone("US/Central")
def scrape_laundry_summary(db: sqlalchemy.engine.base.Engine):
    Og_url = "https://mycscgo.com/laundry/summary/b94db20b-3bf8-4517-9cae-da46d7dd73f6/2303113-025"
    Tr_url = "https://mycscgo.com/laundry/summary/b94db20b-3bf8-4517-9cae-da46d7dd73f6/2303113-026"
    
    try:
        # Navigate to the page
        print("starting scraping...")
        driver.get(Og_url)
        
        # Optionally, wait a few seconds to allow JavaScript to load content (a more robust approach would be using WebDriverWait)
        time.sleep(5)
        
        # Locate the elements by XPath
        washers = '//*[@id="root"]/div/main/div[2]/div/div/div[2]/div[1]/p'
        dryers = '//*[@id="root"]/div/main/div[2]/div/div/div[4]/div[1]/p'
        
        try:
            Og_washers = driver.find_element(By.XPATH, washers).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find element 1: {e}")
        
        try:
            Og_dryers = driver.find_element(By.XPATH, dryers).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find element 2: {e}")

        driver.get(Tr_url)
        
        # Optionally, wait a few seconds to allow JavaScript to load content (a more robust approach would be using WebDriverWait)
        time.sleep(5)
        
        # Locate the elements by XPath
        
        try:
            Tr_washers = driver.find_element(By.XPATH, washers).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find element 1: {e}")
        
        try:
            Tr_dryers = driver.find_element(By.XPATH, dryers).text.split('/')[0].strip()
        except Exception as e:
            print(f"Unable to find element 2: {e}")
    
    finally:
        print("starting SQL")
        # Close the browser
        driver.quit()
        now = datetime.now(tz=tz)
        stmt = sqlalchemy.text(
            "INSERT INTO laundry (washers_available, dryers_available, hall, month, weekday, hour, minute, year, date_added, day) VALUES (:washers, :dryers, :hall, :month, :weekday, :hour, :minute, :year, :date_added, :day)"
        )
        try:
            # Using a with statement ensures that the connection is always released
            # back into the pool at the end of statement (even if an error occurs)
            with db.connect() as conn:
                conn.execute(stmt, parameters={"washers": Og_washers, "dryers": Og_dryers, "hall": 0, "month": now.month, "weekday": now.weekday(), "hour": now.hour, "minute": now.minute, "year": now.year, "date_added": now.strftime("%Y-%m-%d %H:%M:%S"), "day": now.day})
                conn.execute(stmt, parameters={"washers": Tr_washers, "dryers": Tr_dryers, "hall": 1, "month": now.month, "weekday": now.weekday(), "hour": now.hour, "minute": now.minute, "year": now.year, "date_added": now.strftime("%Y-%m-%d %H:%M:%S"), "day": now.day})
                conn.commit()
                print("commited changes")
        except Exception as e:
            # If something goes wrong, handle the error in this section. This might
            # involve retrying or adjusting parameters depending on the situation.
            # [START_EXCLUDE]
            logger.exception(e)
            return