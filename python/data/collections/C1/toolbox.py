import sys
sys.path.append('../../../')
import os
import yfinance as yf
import log
import pandas as pd

logger = log.Log("DEBUG")

def get_ticker(ticker_symbol):
    """Get a ticker object given a ticker symbol"""
    
    logger.INFO(f"Getting ticker {ticker_symbol}...")

    t = yf.Ticker(ticker_symbol)

    try:
        symbol = t.info["symbol"]
    except KeyError:
        logger.ERR(f"Ticker with symbol '{ticker_symbol}' not found")
        raise ValueError(f"Ticker with symbol '{ticker_symbol}' not found")

    if symbol != ticker_symbol:
        logger.ERR(f"Ticker symbols did not match ({t.symbol} != {ticker_symbol})")
        raise ValueError(f"Ticker symbols did not match ({t.symbol} != {ticker_symbol})")

    logger.INFO(f"Ticker found!")

    return t

def get_history_period(t, interval, period):
    """Get the history of a ticker based on interval and period"""

    assert is_ticker_object(t)

    history = t.history(interval=interval, period=period)

    return history

def get_history_start_stop(t, interval, start, stop):
    """Get the history of a ticker based on interval and start/stop dates"""

    assert is_ticker_object(t)

    raise NotImplementedError()

def get_history_period_multiple(ticker_symbols, interval, period):
    """Get history for multiple tickers simultaneously based on interval and period"""
    data = yf.download(ticker_symbols, interval=interval, period=period)

    return data

def is_ticker_object(t):
    """Return true if the supplied object is a yfinance ticker, otherwise false"""
    return type(t) == yf.ticker.Ticker

def mkdir(folder_name):
    """Makes sure a folder exists, otherwise creates it"""

    if not os.path.exists(folder_name):
        logger.INFO(f"Folder did not exist, creating {folder_name}...")
        os.makedirs(folder_name)

def numpy_datetime_to_date(date):
    pandas_datetime = pd.to_datetime(str(date)) 
    date = pandas_datetime.strftime('%Y.%m.%d')
    return date