import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def get_stock_data(company: list, start: str, end: str, features: list = ["Close", "High", "Low", "Open", "Volume"], interval: str = "1d", prepost: bool = False):
    if (_check_company_names(company) and _check_date(start) and _check_date(end) and _check_interval(interval) and _check_features(features)):
        data = yf.download(tickers = company, start = start, end = end, interval = interval)[features]
        
        return data

def _check_date(date):
    try:
        # Ensure the format follows YYYY-MM-DD
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        print(f"You entered {date}. However, the correct syntax for each date (start and end) is as follows: 'yy-mm-dd' (Year-month-day)")
        return False
    
def _check_company_names(company): 
    flag = False
    for i in company:
        try:
            yf.Ticker(i).info["underlyingSymbol"] == i
            flag = True
        except Exception:
            print(f"Company '{i}' does not found. Please be sure that you entered a correct company name.")
            return False
    return True

def _check_interval(interval):
    available_intervals = ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"]
    if (interval in available_intervals):
        return True
    else:
        print(f"Interval is entered as '{interval}', however, the valid intervals are: {available_intervals}")
        return False

def _check_features(features):
    valid_names = {"Close", "High", "Low", "Open", "Volume"}
    
    for i in features:
        if (i not in valid_names):
            print(f"Valid feature names are as follows: {valid_names}. However, entered: {features}")
            return False
    if (len(features) > 5):
        print("There is no repetition allowed in the features. All the feature names must be written only once.")
        return False
    
    return True



