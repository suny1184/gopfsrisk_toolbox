# exploratory data analysis
import pandas as pd
import math
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

# define function to log df info
def LOG_DF_INFO(df, str_dflogname='df_train', str_datecol='dtmStampCreation__app', str_bin_target='TARGET__app', 
	            logger=None, bool_low_memory=True):
	# get rows
	int_nrows = df.shape[0]
	# get columns
	int_ncols = df.shape[1]
	# logic
	if bool_low_memory:
		int_n_missing_all = 0
		# iterate through cols
		for a, col in enumerate(df.columns):
			# print message
			print(f'Checking NaN: {a+1}/{int_ncols}')
			# get number missing per col
			int_n_missing_col = df[col].isnull().sum()
			# add to int_n_missing_all
			int_n_missing_all += int_n_missing_col
		# get proportion NaN
		flt_prop_nan = int_n_missing_all/(int_nrows*int_ncols)
	else:
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

# define function to save proportion NaN by column
def SAVE_NAN_BY_COL(df, str_filename='./output/df_propna.csv', logger=None, bool_low_memory=True):
	# logic
	if bool_low_memory:
		# empty list
		list_empty = []
		# iterate through cols
		for a, col in enumerate(df.columns):
			# print message
			print(f'Checking NaN: {a+1}/{df.shape[1]}')
			# get prop missing
			flt_prop_nan = df[col].isnull().sum()/len(df[col])
			# create dict
			dict_ = {'column': col,
			         'prop_nan': flt_prop_nan}
			# append to list_empty
			list_empty.append(dict_)
		# make df
		df = pd.DataFrame(list_empty)
	else:
		# get proportion missing by col
		ser_propna = df.isnull().sum()/df.shape[0]
		# put into df
		df = pd.DataFrame({'column': ser_propna.index,
	                       'prop_nan': ser_propna})
	# sort
	df.sort_values(by='prop_nan', ascending=False, inplace=True)
	# save to csv
	df.to_csv(str_filename, index=False)
	# if using logger
	if logger:
		logger.warning(f'csv file of proportion NaN by column generated and saved to {str_filename}')

# define function to get training only
def CHRON_GET_TRAIN(df, flt_prop_train=0.5, logger=None):
	# get n_rows in df
	n_rows_df = df.shape[0]
	# get last row in df_train
	n_row_end_train = math.floor(n_rows_df * flt_prop_train)
	# get training data
	df = df.iloc[:n_row_end_train, :]
	# if using logger
	if logger:
		# log it
		logger.warning(f'Subset df to first {flt_prop_train} rows for training')
	# return
	return df

# define class to find/drop features with 100% NaN
class DropAllNaN(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_low_memory=True):
		self.list_cols = list_cols
		self.bool_low_memory = bool_low_memory
	# fit
	def fit(self, X, y=None):
		# logic
		if self.bool_low_memory:
			# empty list
			list_cols_allnan = []
			# iterate through cols
			for a, col in enumerate(X[self.list_cols]):
				# print message
				print(f'Checking NaN: {a+1}/{len(self.list_cols)}')
				# get proportion nan
				flt_prop_nan = X[col].isnull().sum()/X.shape[0]
				# logic
				if flt_prop_nan == 1:
					# append to list
					list_cols_allnan.append(flt_prop_nan)
		else:
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
		for col in self.list_cols_allnan:
			del X[col]
			# print message
			print(f'Dropped {col}')

# define function to log df shape
def LOG_DF_SHAPE(df, logger=None):
	# get rows
	int_nrows = df.shape[0]
	# get columns
	int_ncols = df.shape[1]
	# if logging
	if logger:
		logger.warning(f'df: {int_nrows} rows, {int_ncols} columns')

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
				n_unique = len(pd.value_counts(X[col]))
				# logic to identify no variance cols
				if n_unique == 1:
					list_novar.append(col)
		else:
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
	# transform X
	def transform(self, X):
		# drop list_novar
		for col in self.list_novar:
			del X[col]
			# print message
			print(f'Dropped {col}')

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
	def compare_columns(self, flt_thresh_upper=0.95, tpl_figsize=(10,10), str_dirname='./output/distplots'):
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
				# make distribution plot
				fig, ax = plt.subplots(figsize=tpl_figsize)
				# title
				ax.set_title(f'{col} - Train > Test')
				# plot train
				sns.distplot(df_col['train'], kde=True, color="r", ax=ax)
				# plot valid
				sns.distplot(df_col['valid'], kde=True, color="g", ax=ax)
				# plot test
				sns.distplot(df_col['test'], kde=True, color="b", ax=ax)
				# save plot
				plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
				# close plot
				plt.close()
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
					# make distribution plot
					fig, ax = plt.subplots(figsize=tpl_figsize)
					# title
					ax.set_title(f'{col} Valid > Train')
					# plot train
					sns.distplot(df_col['train'], kde=True, color="r", ax=ax)
					# plot valid
					sns.distplot(df_col['valid'], kde=True, color="g", ax=ax)
					# plot test
					sns.distplot(df_col['test'], kde=True, color="b", ax=ax)
					# save plot
					plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
					# close plot
					plt.close()
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
				# make distribution plot
				fig, ax = plt.subplots(figsize=tpl_figsize)
				# title
				ax.set_title(f'{col} - Train > Test')
				# plot train
				sns.distplot(df_col['train'], kde=True, color="r", ax=ax)
				# plot valid
				sns.distplot(df_col['valid'], kde=True, color="g", ax=ax)
				# plot test
				sns.distplot(df_col['test'], kde=True, color="b", ax=ax)
				# save plot
				plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
				# close plot
				plt.close()
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
					# make distribution plot
					fig, ax = plt.subplots(figsize=tpl_figsize)
					# title
					ax.set_title(f'{col} - Test > Train')
					# plot train
					sns.distplot(df_col['train'], kde=True, color="r", ax=ax)
					# plot valid
					sns.distplot(df_col['valid'], kde=True, color="g", ax=ax)
					# plot test
					sns.distplot(df_col['test'], kde=True, color="b", ax=ax)
					# save plot
					plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
					# close plot
					plt.close()
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
				# make distribution plot
				fig, ax = plt.subplots(figsize=tpl_figsize)
				# title
				ax.set_title(f'{col} - Valid > Test')
				# plot train
				sns.distplot(df_col['train'], kde=True, color="r", ax=ax)
				# plot valid
				sns.distplot(df_col['valid'], kde=True, color="g", ax=ax)
				# plot test
				sns.distplot(df_col['test'], kde=True, color="b", ax=ax)
				# save plot
				plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
				# close plot
				plt.close()
			else:
				# second test (test > valid)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['test'] > x['valid'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between test and valid')
					# append to list
					list_sig_diff.append(col)
					# make distribution plot
					fig, ax = plt.subplots(figsize=tpl_figsize)
					# title
					ax.set_title(f'{col} - Test > Valid')
					# plot train
					sns.distplot(df_col['train'], kde=True, color="r", ax=ax)
					# plot valid
					sns.distplot(df_col['valid'], kde=True, color="g", ax=ax)
					# plot test
					sns.distplot(df_col['test'], kde=True, color="b", ax=ax)
					# save plot
					plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
					# close plot
					plt.close()	
		# save to object
		self.list_sig_diff = list_sig_diff
	# drop columns
	def get_list_good_cols(self):
		# remove the significantly different columns
		list_good_cols = [col for col in self.df_train_sub.columns if col not in self.list_sig_diff]
		# return
		return list_good_cols
