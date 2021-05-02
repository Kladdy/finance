import yfinance as yf
import log

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

def get_history_start_stop(t, inteval, start, stop):
    """Get the history of a ticker based on interval and start/stop dates"""

    assert is_ticker_object(t)

    raise NotImplementedError()

def is_ticker_object(t):
    """Return true if the supplied object is a yfinance ticker, otherwise false"""
    return type(t) == yf.ticker.Ticker
