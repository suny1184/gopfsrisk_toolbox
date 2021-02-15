# model eval
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from itertools import chain
from sklearn.metrics import (accuracy_score, fowlkes_mallows_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss, precision_recall_curve, auc,
	                         roc_curve)
from scipy.stats import zscore
from .general import GET_NUMERIC_AND_NONNUMERIC
from .algorithms import FIT_CATBOOST_MODEL

# define function to get binary eval metrics
def BIN_CLASS_EVAL_METRICS(model_classifier, X, y, logger=None):
	# generate predicted class
	y_hat_class = model_classifier.predict(X)
	# generate predicted probabilities
	y_hat_proba = model_classifier.predict_proba(X)[:,1]
	# metrics
	# accuracy
	accuracy = accuracy_score(y_true=y, y_pred=y_hat_class)
	# geometric mean
	geometric_mean = fowlkes_mallows_score(labels_true=y, labels_pred=y_hat_class)
	# precision
	precision = precision_score(y_true=y, y_pred=y_hat_class)
	# recall
	recall = recall_score(y_true=y, y_pred=y_hat_class)
	# f1
	f1 = f1_score(y_true=y, y_pred=y_hat_class)
	# roc auc
	roc_auc = roc_auc_score(y_true=y, y_score=y_hat_proba)
	# precision recall auc
	pr_auc = average_precision_score(y_true=y, y_score=y_hat_proba)
	# log loss
	log_loss_ = log_loss(y_true=y, y_pred=y_hat_proba)
	# brier 
	brier = brier_score_loss(y_true=y, y_prob=y_hat_proba)
	# put into dictionary
	dict_ = {'accuracy': accuracy,
	         'geometric_mean': geometric_mean,
	         'precision': precision,
	         'recall': recall,
	         'f1': f1,
	         'roc_auc': roc_auc,
	         'pr_auc': pr_auc,
	         'log_loss': log_loss_,
	         'brier': brier}
	# if using logger
	if logger:
		logger.warning('Dictionary of binary eval metrics generated')
	# return dict_
	return dict_

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
	# try saving it
	try:
		# save it
		df_imp.to_csv(str_filename, index=False)
	except FileNotFoundError:
		# make output directory
		os.mkdir('./output')
		# save it
		df_imp.to_csv(str_filename, index=False)
	# if using logger
	if logger:
		# log it
		logger.warning(f'Feature importance saved to {str_filename}')

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

# define function for PR Curve
def PR_CURVE(y_true, y_hat_prob, y_hat_class, tpl_figsize=(10,10), logger=None, str_filename='./output/plt_prcurve.png'):
	# get precision rate and recall rate
	precision_r, recall_r, thresholds = precision_recall_curve(y_true, y_hat_prob)
	# get F1
	flt_f1 = f1_score(y_true, y_hat_class)
	# get auc
	flt_auc = auc(recall_r, precision_r)
	# get the value of chance (i.e., deliquency rate)
	flt_chance = np.sum(y_true)/len(y_true)
	# create ax
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title(f'PR Curve: F1 = {flt_f1:0.4}; AUC = {flt_auc:0.4}')
	# xlabel
	ax.set_xlabel('Recall')
	# ylabel
	ax.set_ylabel('Precision')
	# pr curve
	ax.plot(recall_r, precision_r, label='PR Curve')
	# chance
	ax.plot(recall_r, [flt_chance for x in recall_r], color='red', linestyle=':', label='Chance')
	# legend
	ax.legend()
	# save fig
	plt.savefig(str_filename, bbox_inches='tight')
	# log for logging
	if logger:
		# log it
		logger.warning(f'Precsion-recall curve saved to {str_filename}')
	# return
	return fig

# function for ROC curves
def ROC_AUC_CURVE(y_true, y_hat, tpl_figsize=(10,10), logger=None, str_filename='.output/plt_rocauc.png'):
	# get roc auc
	auc = roc_auc_score(y_true=y_true,
		                y_score=y_hat)
	# get false positive rate, true positive rate
	fpr, tpr, thresholds = roc_curve(y_true=y_true, 
		                             y_score=y_hat)
	# set up subplots
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# set title
	ax.set_title('ROC Plot - (AUC: {0:0.4f})'.format(auc))
    # set x axis label
	ax.set_xlabel('False Positive Rate (Sensitivity)')
    # set y axis label
	ax.set_ylabel('False Negative Rate (1 - Specificity)')
    # set x lim
	ax.set_xlim([0,1])
    # set y lim
	ax.set_ylim([0,1])
    # create curve
	ax.plot(fpr, tpr, label='Model')
    # plot diagonal red, dotted line
	ax.plot([0,1], [0,1], color='red', linestyle=':', label='Chance')
    # create legend
	ax.legend(loc='lower right')
	# fix overlap
	plt.tight_layout()
	# save fig
	plt.savefig(str_filename, bbox_inches='tight')
	# if using logger
	if logger:
		# log it
		logger.warning(f'ROC AUC curve saved in {str_filename}')
	# return fig
	return fig

# define function for residual plot
def RESIDUAL_PLOT(arr_yhat, ser_actual, str_filename, tpl_figsize=(10,10), logger=None):
	# get residuals
	ser_residuals = arr_yhat - ser_actual
	# create ax
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title('Residual Plot (predicted - actual)')
	# distplot
	sns.distplot(ser_residuals, ax=ax)
	# save fig
	plt.savefig(str_filename, bbox_inches='tight')
	# if using logger
	if logger:
		# log it
		logger.warning(f'residual plot saved to {str_filename}')
	# return
	return fig

# define function to programmatically get sensitivity info
def GET_SENSITIVITY_INFO(str_filename='../../09_feature_selection/aaron/output/df_feats.csv', logger=None):
	# load str_filename
	df_feats = pd.read_csv(str_filename)
	# subset to only sensitivity
	df_feats = df_feats[df_feats['analysis_type']=='sensitivity']
	# get max counter
	int_counter_max = np.max(df_feats['counter'])
	# subset to int_counter_max
	df_feats = df_feats[df_feats['counter']==int_counter_max]
	# sort df_feats
	df_feats = df_feats.sort_values(by='eval_metric', ascending=True, inplace=False)
	# make sure each list is a list not a string
	df_feats['list_feats'] = df_feats['list_feats'].apply(lambda x: literal_eval(x))
	# create list of feats
	list_list_feats = list(df_feats['list_feats'])    
	# get list of all features
	list_all_feats = list(chain(*list_list_feats))
	# remove duplicates
	list_all_feats_no_dups = list(set(list_all_feats))
	# get the removed feat
	df_feats['removed_feat'] = df_feats['list_feats'].apply(lambda x: [feat for feat in list_all_feats_no_dups if feat not in x][0])
	# subset to only the columns we will need
	df_feats = df_feats[['removed_feat','eval_metric']]
	# if using logger
	if logger:
		# log it
		logger.warning(f'Sensitivity analysis data imported from {str_filename}')
	# return df_feats
	return df_feats

# define function for sensitivity plot
def SENSITIVITY_PLOT(df, str_eval_metric='PR-AUC', str_filename='./output/plt_sensitivity.png', logger=None):
	# get min
	flt_min_eval_metric = np.min(df_feats['eval_metric'])
	# get max
	flt_max_eval_metric = np.max(df_feats['eval_metric'])
	# get median
	flt_mdn_eval_metric = np.median(df_feats['eval_metric'])
	# create axis
	fig, ax = plt.subplots(figsize=(df_feats.shape[0], 10))
	# title
	ax.set_title(f'{str_eval_metric} (ascending) by Removed Feature')
	# xlabel
	ax.set_xlabel('Removed Feature')
	# ylabel
	ax.set_ylabel(str_eval_metric)
	# line plot of sensitivity analysis
	ax.plot(df_feats['removed_feat'], df_feats['eval_metric'], color='red', label=str_eval_metric)
	# line of min
	ax.plot(df_feats['removed_feat'], [flt_min_eval_metric for x in df_feats['eval_metric']], linestyle=':', color='blue', label=f'Minimum {str_eval_metric} ({flt_min_eval_metric:0.4})')
	# line of max
	ax.plot(df_feats['removed_feat'], [flt_max_eval_metric for x in df_feats['eval_metric']], linestyle=':', color='red', label=f'Maximum {str_eval_metric} ({flt_max_eval_metric:0.4})')
	# line of median
	ax.plot(df_feats['removed_feat'], [flt_mdn_eval_metric for x in df_feats['eval_metric']], linestyle=':', color='green', label=f'Median {str_eval_metric} ({flt_mdn_eval_metric:0.4})')
	# rotate xticks 90 degrees
	plt.xticks(rotation=90)
	# legend
	plt.legend()
	# save figure
	plt.savefig(str_filename, bbox_inches='tight')
	# if using logger
	if logger:
		# log it
		logger.warning(f'Sensitivity analysis plot saved to {str_filename}')
	# return
	return fig

# define function for pd plots
def PARTIAL_DEPENDENCE_PLOTS(model, X_train, y_train, list_cols, tpl_figsize=(15,10), 
	                         str_dirname='./output/pd_plots', str_filename='./output/df_trends.csv', logger=None):
	# generate predicted probabilities
	y_hat_train = model.predict_proba(X_train)[:,1]
	# create dataframe
	X_train['predicted'] = y_hat_train
	X_train['actual'] = y_train
	# create empty df
	df_empty = pd.DataFrame()
	# generate plots
	for a, col in enumerate(list_cols):
		# print meessage
		print(f'Creating plot {a+1}/{len(list_cols)}')
		# group df
		X_train_grouped = X_train.groupby(by=col, as_index=False).agg({'predicted': 'mean',
		                                                               'actual': 'mean'})
		# sort
		X_train_grouped = X_train_grouped.sort_values(by=col, ascending=True)
		# make z score col name
		str_z_col = f'{col}_z'

		# get z score
		X_train_grouped[str_z_col] = zscore(X_train_grouped[col])
		# subset to only those with z >= 3 and <= -3 (i.e., remove outliers)
		X_train_grouped = X_train_grouped[(X_train_grouped[str_z_col] < 3) & (X_train_grouped[str_z_col] > -3)]

		# calculate trendlines
		# predicted
		z_pred = np.polyfit(X_train_grouped[col], X_train_grouped['predicted'], 1)
		p_pred = np.poly1d(z_pred)

		# actual
		z_act = np.polyfit(X_train_grouped[col], X_train_grouped['actual'], 1)
		p_act = np.poly1d(z_act)

		# create predicted array train
		arr_trend_pred = p_pred(X_train_grouped[col])
		# create array for actual
		arr_trend_actual = p_act(X_train_grouped[col])

		# calculate run
		run_ = max_X_train_grouped - min_X_train_grouped

		# calculate slope predicted
		flt_trend_pred = (arr_trend_pred[-1] - arr_trend_pred[0] / run_
		# calculate slope actual
		flt_trend_actual = (arr_trend_actual[-1] - arr_trend_actual[0] / run_

		# make dictionary
		dict_ = {'feature':col, 'trend_pred':flt_trend_pred, 'trend_act':flt_trend_actual}
		# append to df_empty
		df_empty = df_empty.append(dict_, ignore_index=True)
		# write to csv
		df_empty.to_csv(str_filename, index=False)

		# create ax
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# plot trendline
		# predicted
		ax.plot(X_train_grouped[col], arr_trend_pred, color='green', label=f'Trend - Predicted ({flt_trend_pred:0.4})')
		# actual
		ax.plot(X_train_grouped[col], arr_trend_actual, color='orange', label=f'Trend - Actual ({flt_trend_actual:0.4})')
		# plot it
		ax.set_title(col)
		# predicted
		ax.plot(X_train_grouped[col], X_train_grouped['predicted'], color='blue', label='Predicted')
		# actual
		ax.plot(X_train_grouped[col], X_train_grouped['actual'], color='red', linestyle=':', label='Actual')
		# legend
		ax.legend(loc='upper right')
		# save fig
		plt.savefig(f'{str_dirname}/{col}.png', bbox_inches='tight')
		# close plot
		plt.close()
	# delete the predicted and actual columns
	del X_train['predicted'], X_train['actual']
	# if logging
	if logger:
		logger.warning(f'Predicted and actual trends generated and saved to {str_filename}')
		logger.warning(f'{len(list_cols)} partial dependence plots generated and saved to {str_dirname}')

# define function for sensitivity analysis
def SENSITIVITY_ANALYSIS(X_train, X_valid, y_train, y_valid, list_cols, list_class_weights=None, 
	                     str_filename_df='./output/df_sensitivity.csv', str_eval_metric='F1',
	                     logger=None, int_iterations=1000, int_early_stopping_rounds=100,
	                     str_task_type='GPU', bool_classifier=True): 
	# create empty df
	df_sensitivity = pd.DataFrame(columns=['feature_removed', str_eval_metric]) 
	# iterate through columns in X_train
	for a, col in enumerate(list_cols):
		# print message
		print(f'Feature {a+1}/{len(list_cols)}')
		# create list of columns
		list_columns = [x for x in list_cols if x != col]
		# get non-numeric feats
		list_non_numeric = GET_NUMERIC_AND_NONNUMERIC(df=X_train, 
			                                          list_columns=list_columns, 
			                                          logger=None)[1]
		# fit cb model
		model = FIT_CATBOOST_MODEL(X_train=X_train[list_columns],
	                       		   y_train=y_train,
	                       		   X_valid=X_valid[list_columns],
	                       		   y_valid=y_valid,
	                       		   list_non_numeric=list_non_numeric,
	                       		   int_iterations=int_iterations,
	                       		   str_eval_metric=str_eval_metric,
	                       		   int_early_stopping_rounds=int_early_stopping_rounds,
	                       		   str_task_type=str_task_type,
	                       		   bool_classifier=bool_classifier,
	                       		   list_class_weights=list_class_weights)
		# get eval metric
		metric_ = BIN_CLASS_EVAL_METRICS(model_classifier=model,
	                                     X=X_valid[list_columns],
	                                     y=y_valid).get(str_eval_metric.lower()) # this could cause bugs in the future
		# create dictionary
		dict_ = {'feature_removed':col, str_eval_metric:metric_}
		# append dict_ to df_sensitivity
		df_sensitivity = df_sensitivity.append(dict_, ignore_index=True)
		# sort it
		df_sensitivity.sort_values(by=str_eval_metric, ascending=True, inplace=True)
		# write to csv
		df_sensitivity.to_csv(str_filename_df, index=False)
	# if using logger
	if logger:
		logger.warning(f'Sensitivity analysis complete, output saved to {str_filename_df}')

"""
# define function for sensitivity plot
def SENSITIVITY_PLOT(x, y, int_n_feats=30, str_filename='./img/plt_sensitivity.png'):
	# create x_new
	x_new = x[:int_n_feats]
	# create y_new
	y_new = y[:int_n_feats]

	# create axis
	fig, ax = plt.subplots(figsize=(int_n_feats,10))
	# title
	ax.set_title(f'F-1 by Removed Feature: Top {int_n_feats} Only')
	# xlabel
	ax.set_xlabel('Removed Feature')
	# ylabel
	ax.set_ylabel('F-1')

	# line plot of sensitivity analysis
	ax.plot(x_new, y_new, color='red', label='Sensitivity Analysis')
	# line plot of median
	ax.plot(x_new, [np.median(y) for x in y_new], color='black', linestyle=':', label='Median')
	# rotate xticks 90 degrees
	plt.xticks(rotation=90)
	# legend
	plt.legend()
	# save figure
	plt.savefig(str_filename)
	# return
	return fig
"""