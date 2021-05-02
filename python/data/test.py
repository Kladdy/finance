from toolbox import logger, get_ticker, get_history_period

msft = get_ticker("MSFT")
history = get_history_period(msft, "1d", "1m")

print(history)
