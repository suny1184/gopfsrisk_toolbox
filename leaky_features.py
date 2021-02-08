# functions
import logging
import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

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

# define function for loading list of best features
def GET_LIST_OF_BEST_FEATURES(logger=None, str_filename='../09_fitting_models/aaron/output/list_bestfeats.pkl'):
	# load in list of features
	with open(str_filename, 'rb') as file_in:
		list_features = pickle.load(file_in)
	# if using logger
	if logger:
		# log it
		logger.warning(f'List of features loaded from {str_filename}')
	# return
	return list_features

# define function to read csv
def CSV_TO_DF(logger=None, str_filename='../output_data/df_raw_fept1.csv', list_usecols=None, list_parse_dates=None):
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

# define function for splitting into X and y
def X_Y_SPLIT(df_train, df_valid, logger=None, str_targetname='TARGET__app'):
	# train
	y_train = df_train[str_targetname]
	df_train.drop(str_targetname, axis=1, inplace=True)
	# valid
	y_valid = df_valid[str_targetname]
	df_valid.drop(str_targetname, axis=1, inplace=True)
	# if using logger
	if logger:
		# log it
		logger.warning('Train and valid dfs split into X and y')
	# return
	return df_train, y_train, df_valid, y_valid

# define function for computing list of class weights
def GET_LIST_CLASS_WEIGHTS(y_train, logger=None):
	# get list of class weights
	list_class_weights = list(compute_class_weight(class_weight='balanced', 
	                                               classes=np.unique(y_train), 
	                                               y=y_train))

	# if using logger
	if logger:
		# log it
		logger.warning('List of class weights computed')
	# return
	return list_class_weights

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

# define function to get feat imp
def SAVE_FEAT_IMP(model, str_filename='./output/df_featimp.csv', logger=None):
	# get model features
	list_model_features = model.feature_names_
	# get importance
	list_feature_importance = list(model.feature_importances_)
	# put in df
	df_imp = pd.DataFrame({'feature': list_model_features,
	                       'importance': list_feature_importance})
	# sort descending
	df_imp.sort_values(by='importance', ascending=False, inplace=True)
	# save it
	df_imp.to_csv(str_filename, index=False)
	# if using logger
	if logger:
		# log it
		logger.warning(f'feature importance saved to {str_filename}')

# define function for writing to pickle
def PICKLE_TO_FILE(item_to_pickle, str_filename='./output/transformer.pkl', logger=None):
	# pickle file
	pickle.dump(item_to_pickle, open(str_filename, 'wb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')