# functions
import logging
import os
import pickle
import time
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.special import expit
import pandas as pd
from prestige.algorithms import fit_catboost_model
import ast
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

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

# define function to log df shape
def LOG_DF_SHAPE(df, logger=None):
	# get rows
	int_nrows = df.shape[0]
	# get columns
	int_ncols = df.shape[1]
	# if logging
	if logger:
		logger.warning(f'df: {int_nrows} rows, {int_ncols} columns')

# define function for columns to keep
def GET_COLS_TO_KEEP(df, list_bad_strings, logger=None):
	# instantiate empty list
	list_col_keep = []
	# iterate through column names
	for col in df.columns:
		# lower
		col_lower = col.lower()
		# iterate through list_bad_strings
		counter = 0
		for str_ in list_bad_strings:
			if str_ in col_lower:
				counter += 1
		if counter == 0:
			list_col_keep.append(col)
	# if using logger
	if logger:
		logger.warning(f'{len(list_col_keep)} columns to keep for feature selection')
	# return
	return list_col_keep

# define function to subset data frames
def SUBSET_TRAIN_VALID(df_train, df_valid, list_columns, logger=None):
	# subset train
	df_train = df_train[list_columns]
	# subset valid
	df_valid = df_valid[list_columns]
	# if using logger
	if logger:
		logger.warning(f'Train and validation dfs subset to {len(list_columns)} features')
	# return
	return df_train, df_valid

# define function to split into X and y
def SPLIT_TRAIN_VALID_X_Y(df_train, df_valid, str_target='TARGET', logger=None):
	# train
	y_train = df_train[str_target]
	df_train.drop(str_target, axis=1, inplace=True)
	# valid
	y_valid = df_valid[str_target]
	df_valid.drop(str_target, axis=1, inplace=True)
	# if using logger
	if logger:
		logger.warning(f'Split df_train and df_valid into X (features) and y ({str_target})')
	# return
	return df_train, y_train, df_valid, y_valid

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

# define function for computing list of class weights
def GET_LIST_CLASS_WEIGHTS(y_train, logger=None):
	# get list of class weights
	list_class_weights = list(compute_class_weight(class_weight='balanced', 
	                                               classes=np.unique(y_train), 
	                                               y=y_train))

	# if using logger
	if logger:
		# log it
		logger.warning(f'List of class weights {list_class_weights} computed')
	# return
	return list_class_weights

# define pr-AUC custom eval metric
class PrecisionRecallAUC:
	# define a static method to use in evaluate method
	@staticmethod
	def get_pr_auc(y_true, y_pred):
		# fit predictions to logistic sigmoid function
		y_pred = expit(y_pred).astype(float)
		# actual values should be 1 or 0 integers
		y_true = y_true.astype(int)
		# calculate average precision
		flt_pr_auc = average_precision_score(y_true=y_true, y_score=y_pred)
		# return flt_pr_auc
		return flt_pr_auc
	# define a function to tell catboost that greater is better (or not)
	def is_max_optimal(self):
		# greater is better
		return True
	# get the score
	def evaluate(self, approxes, target, weight):
		# make sure length of approxes == 1
		assert len(approxes) == 1
		# make sure length of target is the same as predictions
		assert len(target) == len(approxes[0])
		# set target to integer and save as y_true
		y_true = np.array(target).astype(int)
		# save predictions
		y_pred = approxes[0]
		# generate score
		score = self.get_pr_auc(y_true=y_true, y_pred=y_pred)
		# return the prediction and the calculated weight
		return score, 1
	# get the final score
	def get_final_error(self, error, weight):
		# return error
		return error

# define function for iterative feature selection
def ITERATIVE_FEAT_SELECTION(X_train, y_train, X_valid, y_valid, list_non_numeric, 
							 list_class_weights, int_n_models=50,
							 int_iterations=1000, int_early_stopping_rounds=100,
							 str_eval_metric='F1', int_random_state=42,
							 str_filename='./output/list_bestfeats.pkl',
							 logger=None):
	# instantiate empty list
	list_empty = []
	# build n models
	for a in range(int_n_models):
		# print message
		print(f'Fitting model {a+1}/{int_n_models}')
		# fit model
		model = fit_catboost_model(X_train=X_train, 
								   y_train=y_train, 
					               X_valid=X_valid, 
					               y_valid=y_valid, 
					               list_non_numeric=list_non_numeric, 
					               int_iterations=int_iterations, 
					               str_eval_metric=str_eval_metric, 
					               int_early_stopping_rounds=int_early_stopping_rounds, 
					               str_task_type='GPU', 
					               bool_classifier=True,
					               list_class_weights=list_class_weights,
					               int_random_state=int_random_state)
		# get model features
		list_model_features = model.feature_names_
		# get importance
		list_feature_importance = list(model.feature_importances_)
		# put in df
		df_imp = pd.DataFrame({'feature': list_model_features,
		                       'importance': list_feature_importance})
		# subset to > 0
		list_imp_feats = list(df_imp[df_imp['importance']>0]['feature'])
		# append new feats to list_empty
		for feat in list_imp_feats:
			if feat not in list_empty:
				list_empty.append(feat)
		# pickle list
		pickle.dump(list_empty, open(str_filename, 'wb'))
	# if using logger
	if logger:
		logger.warning(f'Completed iterative feature selection after fitting {int_n_models} models.')
		logger.warning(f'List of features pickled to {str_filename}')