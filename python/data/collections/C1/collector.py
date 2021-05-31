import sys
#sys.path.append('../../')
from toolbox import logger, mkdir, get_ticker, get_history_period_multiple, numpy_datetime_to_date
import numpy as np
import csv

# Constants
collector_name = "C1"
saved_data_dir = "saved_data"
period = "1mo"
interval = "90m"

# Make sure saved_data folder exists

mkdir("saved_data")

ticker_symbol_list = []

# Read index csv to get ticker names
with open('C1_index_components.csv', newline="", encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            ticker_symbol_list.append(row[0]) # Append ticker symbol

# Amount of tickers
N = len(ticker_symbol_list)

# Fetch ticker data
ticker_symbols = " ".join(ticker_symbol_list)

data = get_history_period_multiple(ticker_symbols, interval, period)

dates = data.index.values
start_datetime = dates[0]
end_datetime = dates[-1]

start_date_str = numpy_datetime_to_date(start_datetime)
end_date_str = numpy_datetime_to_date(end_datetime)

# Save to .csv
csv_file_name = f"{collector_name}_period{period}_inteval{interval}_start{start_date_str}_end{end_date_str}.csv"
data.to_csv(f"{saved_data_dir}/{csv_file_name}")

# Save to .pkl
pkl_file_name = f"{collector_name}_period{period}_inteval{interval}_start{start_date_str}_end{end_date_str}.pkl"
data.to_pickle(f"{saved_data_dir}/{pkl_file_name}")

# # Iterate over each ticker
# #for i in range(N):


# for i in range(1):
#     ticker_symbol = ticker_symbol_list[i]

#     # Get the ticker
#     t = get_ticker(ticker_symbol)

#     # Get history
#     hist = get_history_period(t, "1d", "1mo")

#     dates = hist.index.values
#     closing_values = hist["Close"].values
    
#     print(closing_values)
#     print(dates)



