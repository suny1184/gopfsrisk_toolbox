# preprocessing
import logging
import os
import pandas as pd
import math
import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype
import pickle
import math
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# class for min max scaling
class MyMinMaxScaler(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# define fit
	def fit(self, X, y=None):
		# initialize class
		cls_minmaxscaler = MinMaxScaler()
		# fit
		cls_minmaxscaler.fit(X[self.list_cols])
		# save to self
		self.cls_minmaxscaler = cls_minmaxscaler
		# return
		return self
	# define transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()
		# transform
		X_scaled = pd.DataFrame(self.cls_minmaxscaler.transform(X[self.list_cols]), columns=self.list_cols)
		# recreate X
		X[self.list_cols] = X_scaled[self.list_cols]
		# get time
		flt_time = time.perf_counter()-time_start
		# print time
		print(f'Time to scale: {flt_time:0.5} sec.')
		# save to object
		self.flt_time = flt_time
		# return X
		return X

# replace negatives, inf, and -inf with 0
class ReplaceNegativesAndInf(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()
		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# no need to future proof
		X[list_cols] = X[list_cols].mask(X[list_cols] < 0).replace(np.inf, np.nan).fillna(0)
		# get time
		flt_time = time.perf_counter()-time_start
		# print time
		print(f'Time to replace negatives and inf: {flt_time:0.5} sec.')
		# save to object
		self.flt_time = flt_time
		# return
		return X

# rounding binner
class RoundBinning(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_round):
		self.dict_round = dict_round
	# fit
	def fit(self, X):
		return self
	# transform
	def transform(self, X):
		time_start = time.perf_counter()
		# make copy of dict_round
		dict_round = self.dict_round.copy()
		# get list of keys
		list_keys = list(dict_round.keys())
		# iterate through keys
		for key in list_keys:
			# if a key is not in the data frame
			if key not in list(X.columns):
				# delete key from dictionary
				del dict_round[key]
		# iterate through dictionary
		for key, val in dict_round.items():
			X[key] = val * round(pd.to_numeric(X[key]) / val)
		print(f'Time to bin: {time.perf_counter()-time_start:0.5} sec.')
		# return X
		return X

# define class for quantile binning
class QuantileBinning(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, int_n_bins=10):
		self.list_cols = list_cols
		self.int_n_bins = int_n_bins
	# fit
	def fit(self, X):
		# empty lists
		list_list_bins = []
		list_dict_bin_name = []
		# iterate through list cols
		for col in self.list_cols:
			# get bins
			list_bins = list(pd.qcut(X[col], self.int_n_bins, retbins=True, duplicates='drop'))[1]
			# make bin names
			list_bin_name = []
			for a in range(len(list_bins)):
				# calculate mean
				try:
					bin_name = (list_bins[a] + list_bins[a+1]) / 2
				except IndexError:
					bin_name = list_bins[a]
				# append
				list_bin_name.append(bin_name)
			# create dictionary
			dict_bin_name = dict(enumerate(list_bin_name))
			# append
			list_list_bins.append(list_bins)
			list_dict_bin_name.append(dict_bin_name)
		# zip list_dict_bin_name list_list_bins
		list_dict_bin_name_list_bins = list(zip(list_dict_bin_name, list_list_bins))
		# zip list cols
		dict_quantiles = dict(zip(self.list_cols, list_dict_bin_name_list_bins))
		# save to object
		self.dict_quantiles = dict_quantiles
		# return object
		return self
	# transform
	def transform(self, X):
		# make sure all cols are in X
		time_start = time.perf_counter()
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# iterate through columns
		for col in list_cols:
			# get dictionary and list_bins
			dict_bin_name, list_bins = self.dict_quantiles[col]
			# convert column to bin
			X[col] = np.vectorize(dict_bin_name.get)(np.digitize(X[col], list_bins))
		print(f'Time to bin: {time.perf_counter()-time_start:0.5} sec.')
		# return
		return X

# define feature mapper class
class FeatureValueReplacer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_value_replace):
		self.dict_value_replace = dict_value_replace
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		X.replace(self.dict_value_replace, inplace=True)
		# return
		return X

# define cyclic FE class
class CyclicFeatures(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, str_datecol, bool_drop_datecol=True):
		self.str_datecol = str_datecol
		self.bool_drop_datecol = bool_drop_datecol
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# convert to datetime
		if X[self.str_datecol].dtype == 'int64':
			X[self.str_datecol] = pd.to_datetime(X[self.str_datecol], format='%Y%m%d')
		else:
			X[self.str_datecol] = pd.to_datetime(X[self.str_datecol])
		# MONTH RELATIVE TO YEAR
		# get month of year
		X['month_of_year'] = pd.DatetimeIndex(X[self.str_datecol]).month
		# get sin of month
		X[f'{self.str_datecol}_month_year_sin'] = np.sin((X['month_of_year']-1) * (2*np.pi/12))
		# get cos of month
		X[f'{self.str_datecol}_month_year_cos'] = np.cos((X['month_of_year']-1) * (2*np.pi/12))	
		# DAY RELATIVE TO MONTH
		# get day of month
		X['day_of_month'] = pd.DatetimeIndex(X[self.str_datecol]).day
		# make string of year-month
		X['year_month'] = X[self.str_datecol].apply(lambda x: f'{x.year}-{x.month}')
		# get the days in each month
		X['days_in_month'] = X['year_month'].apply(lambda x: pd.Period(x).days_in_month)
		# get sin of day relative to month
		X[f'{self.str_datecol}_day_month_sin'] = np.sin((X['day_of_month']-1) * (2*np.pi/X['days_in_month']))
		# get cosin of day relative to month
		X[f'{self.str_datecol}_day_month_cos'] = np.cos((X['day_of_month']-1) * (2*np.pi/X['days_in_month']))
		# DAY RELATIVE TO WEEK
		# get day of week (starts at zero so we won't subtract 1 below)
		X['day_of_week'] = pd.DatetimeIndex(X[self.str_datecol]).dayofweek
		# get sin of day relative to week
		X[f'{self.str_datecol}_day_week_sin'] = np.sin((X['day_of_week']) * (2*np.pi/7))
		# get cosin of day relative to month
		X[f'{self.str_datecol}_day_week_cos'] = np.cos((X['day_of_week']) * (2*np.pi/7))
		# DAY RELATIVE TO YEAR
		# get day of year
		X['day_of_year'] = pd.DatetimeIndex(X[self.str_datecol]).dayofyear
		# get year
		X['year'] = pd.DatetimeIndex(X[self.str_datecol]).year
		# get last day of year
		X['last_day_of_year'] = X['year'].apply(lambda x: f'12-31-{x}')
		# get days in year so it works with leap years
		X['days_in_year'] = pd.DatetimeIndex(X['last_day_of_year']).dayofyear
		# get sin of day relative to year
		X[f'{self.str_datecol}_day_year_sin'] = np.sin((X['day_of_year']-1) * (2*np.pi/X['days_in_year']))
		# get cosin of day relative to year
		X[f'{self.str_datecol}_day_year_cos'] = np.cos((X['day_of_year']-1) * (2*np.pi/X['days_in_year']))
		# DROP FEATURES
		if self.bool_drop_datecol:
			X.drop(['month_of_year','day_of_month','year_month','days_in_month',
			        'day_of_week','day_of_year','year','last_day_of_year',
					'days_in_year',self.str_datecol], axis=1, inplace=True)
		else:
			X.drop(['month_of_year','day_of_month','year_month','days_in_month',
			        'day_of_week','day_of_year','year','last_day_of_year',
					'days_in_year'], axis=1, inplace=True)
		# return
		return X

# define string converter
class StringConverter(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# make sure all cols are in X
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# convert to string
		X[list_cols] = X[list_cols].applymap(str)
		# return
		return X

# define function for chronological split
def CHRON_TRAIN_VALID_TEST_SPLIT(df, flt_prop_train=0.5, flt_prop_valid=0.25, logger=None):
	# get n_rows in df
	n_rows_df = df.shape[0]
	# get last row in df_train
	n_row_end_train = math.floor(n_rows_df * flt_prop_train)
	# get last row in df_valid
	n_row_end_valid = math.floor(n_rows_df * (flt_prop_train + flt_prop_valid))
	# create train, valid, test
	df_train = df.iloc[:n_row_end_train, :]
	df_valid = df.iloc[n_row_end_train:n_row_end_valid, :]
	df_test = df.iloc[n_row_end_valid:, :]
	# calculate proportion in test
	flt_prop_test = 1 - (flt_prop_train + flt_prop_valid)
	# if using logger
	if logger:
		# log it
		logger.warning(f'Split df into train ({flt_prop_train}), valid ({flt_prop_valid}), and test ({flt_prop_test})')
	# return
	return df_train, df_valid, df_test

# class for converting boolean to binary
class BooleanToBinary(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, str_datecol=None):
		self.str_datecol = str_datecol
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		if self.str_datecol:
			# save str_datecol as a list
			list_ = list(X[self.str_datecol])
			# drop str_datecol
			del X[self.str_datecol]
			# multiply X by 1
			X = X * 1
			# put list_ into X
			X[self.str_datecol] = list_
		else:
			# multiply X by 1
			X = X * 1
		# return
		return X

# define function to get list of cols with threshold nan
def GET_LIST_THRESHOLD_NAN(df, flt_threshold=0.5, logger=None, bool_low_memory=True):
	# empty list
	list_columns = []
	# logic
	if bool_low_memory:
		for a, col in enumerate(df.columns):
			# print message
			print(f'Evaluating column {a+1}/{df.shape[1]}')
			flt_propna = df[col].isnull().sum() / df.shape[0]
			# logic
			if flt_propna >= flt_threshold:
				list_columns.append(col)
	else:
		# get series of columns and proportion missing
		ser_propna = df.isnull().sum() / df.shape[0]
		# subset to >= flt_threshold
		list_columns = list(ser_propna[ser_propna >= flt_threshold].index)
	# if using logger
	if logger:
		logger.warning(f'{len(list_columns)} columns with prop nan >= {flt_threshold}')
	# return
	return list_columns

# define Binaritizer
class Binaritizer(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit to X
	def fit(self, X, y=None):
		return self
	# transform X
	def transform(self, X):
		# make sure all cols in self.list_columns are in X
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# convert X to binary with cols from list_cols
		X_bin = X[list_cols].notnull()*1
		# convert each col name to __bin and set as cols in X_bin
		X_bin.columns = pd.Series(X_bin.columns).apply(lambda x: '{}__bin'.format(x))
		# make sure X_bin.index == X.index
		X_bin.index = X.index
		# concatenate
		X = pd.concat([X, X_bin], axis=1, sort=False)
		# return
		return X

# define function to get list of non-numeric rvlr cols
def GET_NONNUMERIC_RVLR_COLS(df, logger=None):
	# instantiatre empty list
	list_columns = []
	# iterate through cols in df
	for col in df.columns:
		if (not is_numeric_dtype(df[col])) and ('rvlr' in col.lower()):
			list_columns.append(col)
	# if using logger
	if logger:
		logger.warning(f'{len(list_columns)} non-numeric rvlr features identified')
	# return
	return list_columns

# define class to make proportion R, T, and I from rvlr cols
class ProportionRTIConverter(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# define a helper function for lambda
		def HELPER_PROP_RTI(str_, rti):
			# if string is NaN
			if pd.isnull(str_):
				return np.nan
			# if string is not a string
			elif type(str_) != str:
				return str_
			# else
			else:
				# get length of string
				len_str = len(str_)
				# get number of rti is in str_
				n_rti = str_.count(rti)
				# calculate proportion
				prop_rti = n_rti / len_str
				# return proportion
				return prop_rti
		# make sure cols in self.list_cols are in X
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# iterate through cols
		for a, col in enumerate(list_cols):
			# get series
			series_ = X[col]
			# iterate through R, T, I
			for rti in ['R','T','I']:
				# create new col
				X[f'{col}__prop_{rti.lower()}'] = series_.apply(lambda x: HELPER_PROP_RTI(str_=x, rti=rti))
			# drop col
			X.drop(col, axis=1, inplace=True)
		# return df
		return X

# define class for iterative imputing
class IterativeImputerNumeric(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, cls_estimator=BayesianRidge()):
		self.list_cols = list_cols
		self.cls_estimator = cls_estimator
	# fit
	def fit(self, X, y=None):
		# instantiate class
		cls_iterative_imputer = IterativeImputer(max_iter=10, 
			                                     random_state=42,
			                                     estimator=self.cls_estimator)
		# fit
		cls_iterative_imputer.fit(X[self.list_cols])
		# save to object
		self.cls_iterative_imputer = cls_iterative_imputer
		# return
		return self
	# transform
	def transform(self, X):
		# transform
		X_imputed = pd.DataFrame(self.cls_iterative_imputer.transform(X[self.list_cols]), columns=self.list_cols)
		# match indices
		X_imputed.index = X.index
		# concatenate
		X = pd.concat([X_imputed, X[[col for col in X.columns if col not in self.list_cols]]], axis=1)
		# return
		return X

# create median imputer
class ImputerNumeric(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, metric='median', bool_ignore_neg=True):
		self.list_cols = list_cols
		self.metric = metric
		self.bool_ignore_neg = bool_ignore_neg
	# fit to X
	def fit(self, X, y=None):
		# define function to remove negative values from a series
		def drop_negative(ser_, bool_ignore_neg=True):
			if self.bool_ignore_neg:
				# subset series to those >= 0
				return ser_[ser_ >= 0]
			else:
				return ser_
		# logic to support metric
		if self.metric == 'median':
			# get metric for columns in list_cols
			ser_metric = X[self.list_cols].apply(lambda x: np.nanmedian(drop_negative(ser_=x, bool_ignore_neg=self.bool_ignore_neg)), axis=0)
		elif self.metric == 'mean':
			# get metric for columns in list_cols
			ser_metric = X[self.list_cols].apply(lambda x: np.nanmean(drop_negative(ser_=x, bool_ignore_neg=self.bool_ignore_neg)), axis=0)
		# zip into dictionary
		self.dict_metric_ = dict(zip(ser_metric.index, ser_metric))
		return self
	# transform X
	def transform(self, X):
		# fill the nas with dict_metric_
		X.fillna(value=self.dict_metric_, inplace=True)
		return X

# create string imputer
class ImputerStringNonNumeric(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, str_impute='MISSING'):
		self.list_cols = list_cols
		self.str_impute = str_impute
	# fit to X
	def fit(self, X, y=None):
		# create dictionary
		self.dict_impute = dict(zip(self.list_cols, [self.str_impute for col in self.list_cols]))
		# return
		return self
	# transform
	def transform(self, X):
		X.fillna(self.dict_impute, inplace=True)
		# return
		return X

# create mode imputer
class ImputerMode(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit to X
	def fit(self, X, y=None):
		# define function to get mode for each col
		def get_mode(ser_):
			# get mode
			mode_ = pd.value_counts(ser_).index[0]
			return mode_
		# get the mode for each col
		ser_metric = X[self.list_cols].apply(lambda x: get_mode(ser_=x), axis=0)
		# zip into dictionary
		self.dict_mode = dict(zip(ser_metric.index, ser_metric))
		return self
	# transform X
	def transform(self, X):
		# fill the nas with dict_mode
		X.fillna(value=self.dict_mode, inplace=True)
		return X

# class for one-hot encoding
class MyOneHotEncoder(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
		self.cls_onehotencoder = OneHotEncoder(drop=None)
	# def fit
	def fit(self, X, y=None):
		# instantiate class
		cls_onehotencoder = self.cls_onehotencoder
		# fit
		cls_onehotencoder.fit(X[self.list_cols])
		# save to self
		self.cls_onehotencoder = cls_onehotencoder
		# return
		return self
	# define transform
	def transform(self, X):
		X_hot = pd.DataFrame(self.cls_onehotencoder.transform(X[self.list_cols]).toarray())
		# make sure the indices match
		X_hot.index = X.index
		# concatenate the X and X hot dfs
		X = pd.concat([X, X_hot], axis=1, ignore_index=False)
		# drop list_cols
		X.drop(self.list_cols, axis=1, inplace=True)
		# return
		return X