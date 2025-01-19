from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from datetime import datetime
import pytz

tz = pytz.timezone("US/Central")
def scrape_laundry_summary():
    Og_url = "https://mycscgo.com/laundry/summary/b94db20b-3bf8-4517-9cae-da46d7dd73f6/2303113-025"
    Tr_url = "https://mycscgo.com/laundry/summary/b94db20b-3bf8-4517-9cae-da46d7dd73f6/2303113-026"
    
    # Initialize the WebDriver (Chrome in this example)
    # Make sure you've installed ChromeDriver and have it in your PATH.
    driver = webdriver.Chrome()
    
    try:
        # Navigate to the page
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
        # Close the browser
        driver.quit()
        now = datetime.now(tz=tz)
        return([[int(Og_washers), int(Og_dryers), 0, now.month, now.weekday(), now.hour], [int(Tr_washers), int(Tr_dryers), 1, now.month, now.weekday(), now.hour], now.strftime("%I:%M%p")])
