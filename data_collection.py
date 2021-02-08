# functions
import logging
import os
import pandas as pd
import numpy as np
import pickle
import time
from prestige.db_connection import read_sql_file, query_to_df
from sklearn.base import BaseEstimator, TransformerMixin

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

# define function for querying data
def QUERY_DATA(logger=None, str_filename='./sql/query.sql'):
	# start timer
	time_start = time.perf_counter()
	try:
		# pull account, run date, and other data at the account and month level
		df = query_to_df(str_query=read_sql_file(str_file=str_filename),
		                 server='electra',
		                 database='pfsdb')
		# if using logger
		if logger:
			# log it
			logger.warning(f'Data imported from DB in {(time.perf_counter()-time_start)/60:0.4} min.')
		# return df
		return df
	except:
		# if using logger
		if logger:
			# log it
			logger.error('Error importing from DB.')

# define function for reading files
def READ_TEXT_PIPE_FILE(logger=None, str_filename='Application.txt', list_columns=None, str_encoding=None, int_nrows=None, list_parse_dates=None, dict_dtype=None):
	# import file
	df = pd.read_csv(str_filename, delimiter='|', usecols=list_columns, encoding=str_encoding, nrows=int_nrows, parse_dates=list_parse_dates, dtype=dict_dtype)
	# if using logger
	if logger:
		# log it
		logger.warning(f'Imported {str_filename}')
	# return
	return df

# define class
class GetDictForImport:
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		# get series of dtypes and convert to data frame
		df_dtypes = X[self.list_cols].dtypes.to_frame(name='dtype')
		# make the index a column
		df_dtypes.reset_index(drop=False, inplace=True)
		# define helper function
		def make_smaller_dtype(ser_):
			if ser_=='int64':
				return 'int32'
			elif ser_=='float64':
				return 'float32'
			else:
				return np.nan
		# apply function
		df_dtypes['dtype_new'] = df_dtypes.apply(lambda x: make_smaller_dtype(ser_=x['dtype']), 
											     axis=1)
		# drop na
		df_dtypes.dropna(inplace=True)
		# zip into dictionary
		self.dict_dtypes = dict(zip(df_dtypes['index'], df_dtypes['dtype_new']))
		# return self
		return self

# define function for writing to pickle
def PICKLE_TO_FILE(item_to_pickle, str_filename='./output/transformer.pkl', logger=None):
	# pickle file
	pickle.dump(item_to_pickle, open(str_filename, 'wb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')

# define function for loadin#g from pickle
def LOAD_FROM_PICKLE(logger=None, str_filename='../06_preprocessing/output/dict_imputations.pkl'):
	# import pickled_file
	with open(str_filename, 'rb') as f:
		pickled_file = pickle.load(f)
	# if using logger
	if logger:
		# log it
		logger.warning(f'Imported file from {str_filename}')
	# return
	return pickled_file

# define function to convert true/false columns to 1 0
def CONVERT_BOOL_TO_BINARY(df, str_datecol='dtmStampCreation', logger=None):
    # save str_datecol as a list
    list_ = list(df[str_datecol])
    # drop str_datecol
    df.drop(str_datecol, axis=1, inplace=True)
    # multiply df by 1
    df = df * 1
    # put list_ into df
    df[str_datecol] = list_
    # if using logger
    if logger:
        logger.warning('True and False converted into 1 and 0, respectively')
    # return df
    return df

# define function to check if any bigAccountId's are in list_bigaccountid
def CHECK_IF_ACCOUNTID_IN_TBL(df, list_bigaccountid, logger=None):
	# remove duplicates to speed things up
	ser_bigaccountid = df['bigAccountId'].drop_duplicates(keep='first', inplace=False)
	# convert to set
	set_ser_bigaccountid = set(ser_bigaccountid)

	# convert list_bigaccountid to set
	set_list_bigaccountid = set(list_bigaccountid)

	# get the intersection
	list_commonelements = list(set_ser_bigaccountid.intersection(set_list_bigaccountid))
	# get the number of elements
	n_commonelements = len(list_commonelements)
	# if using logger
	if logger:
		# check to see if any bigAccountId's in ser_bigaccountid are in list_bigaccountid
		logger.warning(f'{n_commonelements} bigAccountIds is/are in tblAccount')

# define function for appending string at end of each col
def APPEND_TO_COL_NAMES(df, str_append='__app', logger=None):
	# append str_append to end of each col
	df.columns = [f'{col}{str_append}' for col in df.columns]
	# if we are using a logger
	if logger:
		logger.warning(f'{str_append} appended to the end of each column name')
	# return
	return df

# define function for creating aggregation dictionary
def CREATE_AGG_DICT(list_numeric, list_non_numeric, logger=None, str_filename='./output/dict_debt_agg.pkl'):
	# create empty dictionary
	dict_empty = {}
	# create aggregation dictionary for numeric cols
	for col in list_numeric:
		dict_empty[col] = ['min','max','sum','mean','median','std', 'count']

	# create empty dictionary
	dict_empty_nonnumeric = {}
	# create aggregation dictionary for non-numeric cols
	for col in list_non_numeric:
		dict_empty_nonnumeric[col] = ['count', pd.Series.nunique]

	# update dict_empty
	dict_empty.update(dict_empty_nonnumeric)

	# pickle dict_empty
	with open(str_filename, 'wb') as file_out:
		pickle.dump(dict_empty, file_out)

	# if we have a logger
	if logger:
		logger.warning(f'Aggregation dictionary pickled to {str_filename}')

	# return
	return dict_empty

# define function to get column names
def GET_COL_NAMES_FROM_DICT_AGG(dict_agg, str_unique_id='UniqueID__debt', logger=None):
	# rename cols
	list_empty = [str_unique_id]
	# iterate through key value pairs in dict_agg
	for key, list_val in dict_agg.items():
		# iterate though value list
		for val in list_val:
			# if val is a string
			if type(val) != str:
				# set val to nunique
				val = 'nunique'
			# create new col name
			str_col_name = f'{key}_{val}'
			# append to list_empty
			list_empty.append(str_col_name)
	# if we have a logger
	if logger:
		logger.warning('Column names from dict_agg created')
	# return
	return list_empty

# define function for columns to keep
def GET_COLS_TO_KEEP_AND_DROP(list_columns_all, list_bad_strings, logger=None):
	# instantiate empty lists
	list_col_keep = []
	list_col_drop = []
	# iterate through column names
	for col in list_columns_all:
		# iterate through list_bad_strings
		counter = 0
		for str_ in list_bad_strings:
			if str_ in col:
				counter += 1
		if counter == 0:
			list_col_keep.append(col)
		else:
			list_col_drop.append(col)
	# if using logger
	if logger:
		logger.warning(f'{len(list_col_keep)} columns to keep, {len(list_col_drop)} columns to drop')
	# return
	return list_col_keep, list_col_drop

# define class to find/drop features with 100% NaN
class DropAllNaN(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		# get proportion missing per column
		ser_propna = X[self.list_cols].isnull().sum()/X.shape[0]
		# subset to 1.0
		list_cols_allnan = list(ser_propna[ser_propna==1.0].index)
		# save into object
		self.list_cols_allnan = list_cols_allnan
		# return object
		return self
	# transform
	def transform(self, X):
		# drop features
		X = X.drop(self.list_cols_allnan, axis=1, inplace=False)
		# return X
		return X

# define class to find/drop features with no variance
class DropNoVariance(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		# define helper function
		def GET_NUNIQUE(ser_):
			n_unique = len(pd.value_counts(ser_))
			return n_unique
		# apply function to every column
		ser_nunique = X[self.list_cols].apply(lambda x: GET_NUNIQUE(ser_=x), axis=0)
		# get the cols with nunique == 1
		list_novar = list(ser_nunique[ser_nunique==1].index)
		# save to object
		self.list_novar = list_novar
		# return self
		return self
	# transform
	def transform(self, X):
		# drop features
		X = X.drop(self.list_novar, axis=1, inplace=False)
		# return X
		return X

# define class
class DropRedundantFeatures(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		# instantiate empty list
		list_redundant_cols = []
		for a, cola in enumerate(self.list_cols):
			# status message
			print(f'Currently, there are {len(list_redundant_cols)} redundant columns.')
			# status message
			print(f'Checking column: {cola}: {a+1}/{len(self.list_cols)}')
			# logic
			if cola not in list_redundant_cols:
				# iterate through the other cols
				for colb in self.list_cols[a+1:]:
					# check if cola == colb
					if X[cola].equals(X[colb]):
						# print message
						print(f'{colb} is redundant with {cola}')
						# append to list_redundant_cols
						list_redundant_cols.append(colb)
		# save to object
		self.list_redundant_cols = list_redundant_cols
		# return
		return self
	# transform
	def transform(self, X):
		# drop list_redundant_cols
		X = X.drop(self.list_redundant_cols, axis=1, inplace=False)
		# return X
		return X