# functions
import logging
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from prestige.algorithms import fit_catboost_model
import matplotlib.pyplot as plt

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

# define function for splitting into X and y
def X_Y_SPLIT(df_train, df_valid, df_test, logger=None, str_targetname='TARGET__app'):
	# train
	y_train = df_train[str_targetname]
	del df_train[str_targetname]
	# valid
	y_valid = df_valid[str_targetname]
	del df_valid[str_targetname]
	# test
	y_test = df_test[str_targetname]
	del df_test[str_targetname]
	# if using logger
	if logger:
		# log it
		logger.warning('Train, valid, and test dfs split into X and y')
	# return
	return df_train, y_train, df_valid, y_valid, df_test, y_test

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

# define function for combining train and valid
def COMBINE_TRAIN_AND_VALID(X_train, X_valid, y_train, y_valid, logger=None):
	# combine train and valid dfs
	X_train = pd.concat([X_train, X_valid])
	y_train = pd.concat([y_train, y_valid])
	# if using logger
	if logger:
		# log it
		logger.warning('Training and validation data combined')
	# return
	return X_train, y_train

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

# define function to fit models iterating through random state
def ITERATIVE_MODEL_FITTING(X_train, y_train, X_valid, y_valid, list_class_weights, list_feats, list_non_numeric, 
	                        int_n_randstate=50, int_iterations=10000, int_early_stopping_rounds=1000,
		                    str_filename='./output/df_randstates.csv', int_randstate_start=0, logger=None,
		                    str_eval_metric='F1', str_task_type='GPU'):
	# create message
	str_message = f'Iterative model fitting for {int_n_randstate} rounds starting at random_state {int_randstate_start}'
	# print it
	print(str_message)
	# if using logger
	if logger:
		# log it
		logger.warning(str_message)

	try:
		# read in str_filename
		df_empty = pd.read_csv(str_filename)
	except FileNotFoundError:
		# create empty df
		df_empty = pd.DataFrame()

	# iterate through random states
	counter = 0
	for int_random_state in range(int_randstate_start, (int_randstate_start+int_n_randstate)):
		# print message
		print(f'Fitting model {int_random_state+1}/{int_n_randstate}')
		# fit cb model
		model = fit_catboost_model(X_train=X_train[list_feats], 
		                           y_train=y_train, 
		                           X_valid=X_valid[list_feats], 
		                           y_valid=y_valid, 
		                           list_non_numeric=list_non_numeric, 
		                           int_iterations=int_iterations, 
		                           str_eval_metric=str_eval_metric, 
		                           int_early_stopping_rounds=int_early_stopping_rounds, 
		                           str_task_type=str_task_type, 
		                           bool_classifier=True,
		                           list_class_weights=list_class_weights,
		                           int_random_state=int_random_state)
		
		# logic
		if str_eval_metric == 'F1':
			# get eval metric
			flt_evalmetric = average_precision_score(y_true=y_valid, y_score=model.predict_proba(X_valid[list_feats])[:,1])
			#flt_evalmetric = f1_score(y_true=y_valid, y_pred=model.predict(X_valid[list_feats]))
		elif str_eval_metric == 'Precision':
			# get eval metric
			flt_evalmetric = precision_score(y_true=y_valid, y_pred=model.predict(X_valid[list_feats]))
		elif str_eval_metric == 'Recall':
			# get eval metric
			flt_eval_metric = recall_score(y_true=y_valid, y_pred=model.predict(X_valid[list_feats]))

		# if we are on first iteration
		if counter == 0:
			# save flt_evalmetric as the current high
			flt_evalmetric_curr_max = flt_evalmetric
			# pickle model
			pickle.dump(model, open('./best_model/cb_model.sav', 'wb'))
		# if we are not on the first iteration and we have a new high eval metric score
		if (counter > 0) and (flt_evalmetric > flt_evalmetric_curr_max):
			# save flt_evalmetric as the current high
			flt_evalmetric_curr_max = flt_evalmetric
			# pickle model
			pickle.dump(model, open('./best_model/cb_model.sav', 'wb'))

		# create dict
		dict_ = {'random_state': int_random_state,
		         'eval_metric': flt_evalmetric,
		         'list_feats': model.feature_names_,
		         'n_feats': len(model.feature_names_),
		         'n_iterations': int_iterations,
				 'n_early_stopping': int_early_stopping_rounds}
		# append to df_empty
		df_empty = df_empty.append(dict_, ignore_index=True)
		# wite to csv
		df_empty.to_csv(str_filename, index=False)
		# increase counter by 1
		counter += 1
	# return
	return df_empty

# define function for plotting df_randstates
def PLOT_DF_RANDSTATES(logger, df_randstates, tpl_figsize=(15,10), str_filename='./output/plt_randstates.png'):
	# find max eval_metric
	flt_max = np.max(df_randstates['eval_metric'])
	# find corresponding random_state
	int_rand_state = df_randstates[df_randstates['eval_metric']==flt_max]['random_state'].iloc[0]

	# create ax
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title(f'Best F1 ({flt_max:0.4}) at Random State {int_rand_state}')
	# plot it
	ax.plot(df_randstates['random_state'], df_randstates['eval_metric'], label='F1')
	# plot the maximum eval metric
	ax.plot(df_randstates['random_state'], [flt_max for x in df_randstates['random_state']], linestyle=':', label='Max F1')
	# xlabel
	ax.set_xlabel('Random State')
	# ylabel
	ax.set_ylabel('F1')
	# legend
	ax.legend()
	# save
	plt.savefig(str_filename, bbox_inches='tight')
	# log it
	logger.warning(f'Plot of df_randstates.csv saved to {str_filename}')
	# return
	return fig