# functions
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

# define function to convert true/false columns to 1 0
def CONVERT_BOOL_TO_BINARY(df, str_datecol='dtmStampCreation__app', logger=None):
	# save str_datecol as a list
	list_ = list(df[str_datecol])
	# drop str_datecol
	del df[str_datecol]
	# multiply df by 1
	df = df * 1
	# put list_ into df
	df[str_datecol] = list_
	# if using logger
	if logger:
		logger.warning('True and False converted into 1 and 0, respectively')
	# return df
	return df

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
	def __init__(self, list_columns, bool_inplace=False):
		self.list_columns = list_columns
		self.bool_inplace = bool_inplace
	# fit to X
	def fit(self, X, y=None):
		return self
	# transform X
	def transform(self, X):
		# convert X to binary with cols from list_cols
		X_bin = X[self.list_columns].notnull()*1
		# if we are replacing
		if self.bool_inplace:
			# drop list_columns from X
			X.drop(self.list_columns, axis=1, inplce=True)
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

# create proportion R, T, and I from rvlr cols
def PROP_RTI(df, list_str_rti_rvlr_col, bool_drop_col=True, logger=None):
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
	# iterate through cols
	for a, col in enumerate(list_str_rti_rvlr_col):
		# if using logger
		if logger:
			logger.warning(f'Converting {col} to proportions of R, T, and I; {a+1}/{len(list_str_rti_rvlr_col)}')
		# get series
		series_ = df[col]
		# iterate through R, T, I
		for rti in ['R','T','I']:
			# create new col
			df[f'{col}__prop_{rti}'] = series_.apply(lambda x: HELPER_PROP_RTI(str_=x, rti=rti))
		# if we want to drop the original col
		if bool_drop_col:
			# drop col
			df.drop(col, axis=1, inplace=True)
			# if using logger
			if logger:
				logger.warning(f'{col} has been dropped from df.')
	# return df
	return df

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

# class for min max scaling
class MyMinMaxScaler(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self):
		self.cls_minmaxscaler = MinMaxScaler()
	# define fit
	def fit(self, X, y=None):
		# instantiate class
		cls_minmaxscaler = self.cls_minmaxscaler
		# fit
		cls_minmaxscaler.fit(X)
		# save to self
		self.cls_minmaxscaler = cls_minmaxscaler
		# return
		return self
	# define transform
	def transform(self, X):
		# transform
		X = pd.DataFrame(self.cls_minmaxscaler.transform(X), columns=X.columns)
		# return X
		return X