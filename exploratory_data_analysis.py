# functions
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle
import time
import numpy as np
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

# define function to log df info
def LOG_DF_INFO(df, str_dflogname='df_train', str_datecol='dtmStampCreation__app', str_bin_target='TARGET__app', logger=None):
	# get rows
	int_nrows = df.shape[0]
	# get columns
	int_ncols = df.shape[1]
	# get proportion NaN
	flt_prop_nan = np.sum(df.isnull().sum())/(int_nrows*int_ncols)
	# get min str_datecol
	min_ = np.min(df[str_datecol])
	# get max dtmstampCreation__app
	max_ = np.max(df[str_datecol])
	# get deliquency rate
	flt_prop_delinquent = np.mean(df[str_bin_target])
	# if logging
	if logger:
		logger.warning(f'{str_dflogname}: {int_nrows} rows, {int_ncols} columns')
		logger.warning(f'{str_dflogname}: {flt_prop_nan:0.4} NaN')
		logger.warning(f'{str_dflogname}: Min {str_datecol} = {min_}')
		logger.warning(f'{str_dflogname}: Max {str_datecol} = {max_}')
		logger.warning(f'{str_dflogname}: Target Proportion = {flt_prop_delinquent:0.4}')

# define DropNoVariance
class DropNoVariance(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, bool_low_memory=True):
		self.list_cols = list_cols
		self.bool_low_memory = bool_low_memory
	# fit to X
	def fit(self, X, y=None):
		# if we have low memory
		if self.bool_low_memory:
			# instantiate empty list
			list_novar = []
			# iterate through cols
			for a, col in enumerate(self.list_cols):
				# print message
				print(f'Checking col {a+1}/{len(self.list_cols)}')
				# get number of unique
				n_unique = len(set(X[col]))
				# logic to identify no variance cols
				if n_unique == 1:
					list_novar.append(col)
		else:
			# define helper function
			def GET_NUNIQUE(ser_):
				n_unique = len(set(ser_))
				#n_unique = len(pd.value_counts(ser_))
				return n_unique
			# apply function to every column
			ser_nunique = X[self.list_cols].apply(lambda x: GET_NUNIQUE(ser_=x), axis=0)
			# get the cols with nunique == 1
			list_novar = list(ser_nunique[ser_nunique==1].index)
		# save to object
		self.list_novar = list_novar
		# return self
		return self
	# transform X
	def transform(self, X):
		# drop list_novar
		for col in self.list_novar:
			del X[col]
			# print message
			print(f'Dropped {col}')
		# return X
		return X

# define class
class DropRedundantFeatures(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, int_n_rows_check=10000):
		self.list_cols = list_cols
		self.int_n_rows_check = int_n_rows_check
	# fit
	def fit(self, X, y=None):
		# instantiate empty list
		list_redundant_cols = []
		for a, cola in enumerate(self.list_cols):
			# status message
			print(f'Currently, there are {len(list_redundant_cols)} redundant columns.')
			# status message
			print(f'Checking column {a+1}/{len(self.list_cols)}')
			# logic
			if cola not in list_redundant_cols:
				# iterate through the other cols
				for colb in self.list_cols[a+1:]:
					# check if subset of cola == colb
					if X[cola].iloc[:self.int_n_rows_check].equals(X[colb].iloc[:self.int_n_rows_check]):
						# print message
						print(f'First {self.int_n_rows_check} rows in {colb} are redundant with {cola}')
						# check if the whole column is redundant
						if X[cola].equals(X[colb]):
							# print message
							print(f'After checking all rows, {colb} is redundant with {cola}')
							list_redundant_cols.append(colb)
						else:
							print(f'After checking all rows, {colb} is not redundant with {cola}')
		# save to object
		self.list_redundant_cols = list_redundant_cols
		# return
		return self
	# transform
	def transform(self, X):
		# drop list_redundant_cols
		for col in self.list_redundant_cols:
			del X[col]
			# print message
			print(f'Dropped {col}')
		# return X
		return X

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

# define class for automating distribution plot analysis
class DistributionAnalysis:
	# initialiaze
	def __init__(self, list_cols, int_nrows=1000, int_random_state=42):
		self.list_cols = list_cols
		self.int_nrows = int_nrows
		self.int_random_state = int_random_state
	# random sample
	def get_random_sample(self, X, str_df_name='train'):
		# logic
		if str_df_name == 'train':
			self.df_train_sub = X.sample(n=self.int_nrows, random_state=self.int_random_state)
		elif str_df_name == 'valid':
			self.df_valid_sub = X.sample(n=self.int_nrows, random_state=self.int_random_state)
		else:
			self.df_test_sub = X.sample(n=self.int_nrows, random_state=self.int_random_state)
	# compare each col
	def compare_columns(self, flt_thresh_upper=0.95):
		# iterate through cols
		list_sig_diff = []
		for a, col in enumerate(self.list_cols):
			# print
			print(f'Currently {len(list_sig_diff)} columns with a significant difference')
			# print
			print(f'Evaluating col {a+1}/{len(self.list_cols)}')
			# create a df with just the cols
			df_col = pd.DataFrame({'train': list(self.df_train_sub[col]),
				                   'valid': list(self.df_valid_sub[col]),
				                   'test': list(self.df_test_sub[col])})
			# TRAIN VS. VALID
			# first test (train > valid)
			flt_avg = np.mean(df_col.apply(lambda x: 1 if x['train'] > x['valid'] else 0, axis=1))
			# logic for significance
			if (flt_avg >= flt_thresh_upper):
				# print
				print(f'Significant difference in {col} between train and valid')
				# append to list
				list_sig_diff.append(col)
				# move to next col
				continue
			else:
				# second test (valid > train)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['valid'] > x['train'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between train and valid')
					# append to list
					list_sig_diff.append(col)
					# move to next col
					continue
			# TRAIN VS. TEST
			# first test (train > test)
			flt_avg = np.mean(df_col.apply(lambda x: 1 if x['train'] > x['test'] else 0, axis=1))
			# logic for significance
			if (flt_avg >= flt_thresh_upper):
				# print
				print(f'Significant difference in {col} between train and test')
				# append to list
				list_sig_diff.append(col)
				# move to next col
				continue
			else:
				# second test (test > train)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['test'] > x['train'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between train and test')
					# append to list
					list_sig_diff.append(col)
					# move to next col
					continue
			# VALID VS. TEST
			# first test (valid > test)
			flt_avg = np.mean(df_col.apply(lambda x: 1 if x['valid'] > x['test'] else 0, axis=1))
			# logic for significance
			if (flt_avg >= flt_thresh_upper):
				# print
				print(f'Significant difference in {col} between valid and test')
				# append to list
				list_sig_diff.append(col)
			else:
				# second test (test > valid)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['test'] > x['valid'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between test and valid')
					# append to list
					list_sig_diff.append(col)	
		# save to object
		self.list_sig_diff = list_sig_diff
	# drop columns
	def get_list_good_cols(self):
		# remove the significantly different columns
		list_good_cols = [col for col in self.df_train_sub.columns if col not in self.list_sig_diff]
		# return
		return list_good_cols

# define function for writing to pickle
def PICKLE_TO_FILE(item_to_pickle, str_filename='./output/transformer.pkl', logger=None):
	# pickle file
	pickle.dump(item_to_pickle, open(str_filename, 'wb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')


"""
# define class for getting medians
class GetMedians:
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
		self.list_df_medians = []
	# calculate medians
	def calculate_medians(self, X):
		# create series of medians
		ser_medians = X[self.list_cols].apply(lambda x: np.median(x), axis=0)
		# save to object
		self.ser_medians = ser_medians
		# return self
		return self
	# create a df
	def create_df(self, str_colname='mdn_train'):
		# create df
		df_medians = pd.DataFrame(self.ser_medians, columns=[str_colname])
		# save to object
		self.df_medians = df_medians
		# return self
		return self
	# append to list of dfs
	def append_df_to_list(self):
		# append to list
		self.list_df_medians.append(self.df_medians)
		# return self
		return self
	# concatenate dfs
	def concatenate_dfs(self):
		# concatenate dfs by index
		df_medians = pd.concat(self.list_df_medians, axis=1)
		# create feature col from index and reset index
		df_medians = df_medians.rename_axis('feature').reset_index()
		# save to object
		self.df_medians = df_medians
		# return self
		return self
	# write to csv
	def write_to_csv(self, str_filename='./output/df_medians.csv'):
		# write to csv
		self.df_medians.to_csv(str_filename, index=False)
"""

"""
# define function to plot 
def DISTPLOTS(df, tpl_figsize=(10,10), logger=None, str_dirname='./output/distplots'):
	# convert df to list of dictionaries
	list_dict_df = df.to_dict(orient='records')
	# iterate through list_dict_df
	for a, dict_ in enumerate(list_dict_df):
		# print message
		print(f'Generating plot {a+1}/{len(list_dict_df)}')
		# create canvas
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# title
		ax.set_title(f'Medians for {dict_["feature"]}')
		# x axis
		ax.set_xlabel('Data Set: Train-Valid-Test')
		# y label
		ax.set_ylabel('Median')
		# generate plot
		ax.bar(['Train', 'Valid', 'Test'], [dict_['mdn_train'], dict_['mdn_valid'], dict_['mdn_test']])
		# save fig
		plt.savefig(f'{str_dirname}/{dict_["feature"]}', bbox_inches='tight')
		# close plot
		plt.close()
		# if using logger
		if logger:
			logger.warning(f'Distribution plots generated and saved to {str_dirname}')
"""