import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import catboost as cb

# define function for fitting catboost model
def FIT_CATBOOST_MODEL(X_train, y_train, X_valid, y_valid, list_non_numeric, int_iterations, str_eval_metric, 
	                   int_early_stopping_rounds, str_task_type='GPU', bool_classifier=True, list_class_weights=None, 
	                   dict_monotone_constraints=None, int_random_state=None):
	# pool data sets
	# train
	train_pool = cb.Pool(X_train, 
	                     y_train,
	                     cat_features=list_non_numeric)
	# valid
	valid_pool = cb.Pool(X_valid,
	                     y_valid,
	                     cat_features=list_non_numeric) 
	# if fitting classifier
	if bool_classifier:
		# instantiate CatBoostClassifier model
		model = cb.CatBoostClassifier(iterations=int_iterations,
		                              eval_metric=str_eval_metric,
		                              task_type=str_task_type,
		                              class_weights=list_class_weights,
		                              monotone_constraints=dict_monotone_constraints,
		                              random_state=int_random_state)
	else:
		# instantiate CatBoostRegressor model
		model = cb.CatBoostRegressor(iterations=int_iterations,
		                             eval_metric=str_eval_metric,
		                             task_type=str_task_type,
		                             monotone_constraints=dict_monotone_constraints,
		                             random_state=int_random_state)
	# fit to training
	model.fit(train_pool,
	          eval_set=[valid_pool], # can only handle one eval set when using gpu
	          use_best_model=True,
	          early_stopping_rounds=int_early_stopping_rounds)
	# return model
	return model