# functions
import logging
import time
import pandas as pd
import pickle
from pandas.api.types import is_numeric_dtype

# define function for logging
def LOG_EVENTS(str_filename='./logs/db_pull.log'):
	# set logging format
	FORMAT = '%(name)s:%(levelname)s:%(asctime)s:%(message)s'
	# get logger
	logger = logging.getLogger(__name__)
	# try making log
	try:
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	except FileNotFoundError:
		os.mkdir('./logs')
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	# change to append
	handler = logging.FileHandler(str_filename, mode='a')
	# set the level to info
	handler.setLevel(logging.INFO)
	# set format
	formatter = logging.Formatter(FORMAT)
	# format the handler
	handler.setFormatter(formatter)
	# add handler
	logger.addHandler(handler)
	# return logger
	return logger

# define function for loadin#g from pickle
def LOAD_FROM_PICKLE(logger=None, str_filename='../06_preprocessing/output/dict_imputations.pkl'):
	# get file
	pickled_file = pickle.load(open(str_filename, 'rb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Imported file from {str_filename}')
	# return
	return pickled_file

# define function to read csv
def CSV_TO_DF(logger=None, str_filename='../output_data/df_raw.csv', list_usecols=None, list_parse_dates=None):
	# start timer
	time_start = time.perf_counter()
	# read json file
	df = pd.read_csv(str_filename, parse_dates=list_parse_dates, usecols=list_usecols)
	# if we are using a logger
	if logger:
		# log it
		logger.warning(f'Data imported from {str_filename} in {(time.perf_counter()-time_start)/60:0.4} min.')
	# return
	return df

# define function to convert date col to datetime and sort
def SORT_DF(df, str_colname='dtmStmpCreation__app', logger=None, bool_dropcol=False):
	# get series of str_colname
	ser_ = df[str_colname]
	# sort ascending
	ser_sorted = ser_.sort_values(ascending=True)
	# get the index as a list
	list_ser_sorted_index = list(ser_sorted.index)
	# order df
	df = df.reindex(list_ser_sorted_index)
	# if dropping str_colname
	if bool_dropcol:
		# drop str_colname
		del df[str_colname]
	# if using a logger
	if logger:
		# log it
		logger.warning(f'df sorted ascending by {str_colname}.')
	# return df
	return df

# define function to get numeric and non-numeric cols
def GET_NUMERIC_AND_NONNUMERIC(df, list_columns, logger=None):
	# instantiate empty lists
	list_numeric = []
	list_non_numeric = []
	# iterate through list_columns
	for col in list_columns:
		# if its numeric
		if is_numeric_dtype(df[col]):
			# append to list_numeric
			list_numeric.append(col)
		else:
			# append to list_non_numeric
			list_non_numeric.append(col)
	# if using logger
	if logger:
		logger.warning(f'{len(list_numeric)} numeric columns identified, {len(list_non_numeric)} non-numeric columns identified')
	# return both lists
	return list_numeric, list_non_numeric

# define function for writing to pickle
def PICKLE_TO_FILE(item_to_pickle, str_filename='./output/transformer.pkl', logger=None):
	# pickle file
	pickle.dump(item_to_pickle, open(str_filename, 'wb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')