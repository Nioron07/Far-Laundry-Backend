import requests
import datetime
import pandas as pd

def get_day_of_week(date_str):
    """Gets the day of the week from a date string.

    Args:
        date_str: A date string in the format YYYY-MM-DD.

    Returns:
        The day of the week as a string (e.g., "Monday", "Tuesday", etc.).
    """

    date_obj = datetime.datetime.strptime(date_str, "%m/%d/%Y").date()
    return date_obj.weekday()

def ConvertTimestamp(dataframe):
    months = []
    daysOfTheWeek = []
    hours = []
    for timestamp in dataframe['Timestamp']:
        months.append(str(timestamp).split("/")[0])
        daysOfTheWeek.append(get_day_of_week(str(timestamp).split(" ")[0]))
        hours.append(str(timestamp).split(" ")[1].split(":")[0])
    
    dataframe['Month'] = months
    dataframe['Weekday'] = daysOfTheWeek
    dataframe['Hour'] = hours

def ConvertHall(dataframe):
    hallDict = {'Oglesby': 0, 'Trelease': 1}
    halls = []
    for hall in dataframe['What Hall?']:
        halls.append(hallDict[hall])
    dataframe['What Hall?'] = halls

def UpdateDataFiles():
    with open("Laundry Data Pro\Laundry Data Pro\\backend\\Data Files\\DirtyData.txt", "w") as file:
        file.write(requests.get("https://docs.google.com/spreadsheets/d/1pWSXxci7Fah_sV5U9hRdXjSkr4hsvlP2EAPg1vPnSaM/gviz/tq?tqx=out:csv&sheet=Form Responses 1").content.decode())
    df = pd.read_csv("Laundry Data Pro\\Laundry Data Pro\\backend\\Data Files\\DirtyData.txt")
    df.dropna(axis=1, inplace=True)

    ConvertTimestamp(df)
    ConvertHall(df)
    df.drop(columns=['Timestamp'], inplace=True)
    df.to_csv("Laundry Data Pro\\Laundry Data Pro\\backend\\Data Files\\GoogleFormData.csv", index=False)