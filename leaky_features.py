# leaky features

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
