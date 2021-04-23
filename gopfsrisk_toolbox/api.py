# api
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from io import StringIO
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# define dropper class
class FinalImputer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_imputations):
		self.dict_imputations = dict_imputations
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# fillna
		X.fillna(self.dict_imputations, inplace=True)
		# return
		return X

# define subsetter class
class Subsetter(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# subset
		X = X[self.list_cols]
		# return
		return X

# define pipeline class
class PipelineDataPrep:
	# initialize
	def __init__(self, list_transformers, model, bool_classifier=True):
		self.list_transformers = list_transformers
		self.model = model
		self.bool_classifier = bool_classifier
	# prep predict
	def prep_predict(self, X):
		# loop through transformers
		for transformer in self.list_transformers:
			# transform
			X = transformer.transform(X)
		# logic
		if self.bool_classifier:
			# make predictions
			y_hat = self.model.predict_proba(X[self.model.feature_names_])[:,1]
		else:
			y_hat = self.model.predict(X[self.model.feature_names_])
		# get mean
		y_hat = np.mean(y_hat)
		# return
		return y_hat

# define class
class ParsePayload:
	# initialize
	def __init__(self, list_feats_raw_app, 
			           list_feats_raw_inc, 
					   list_feats_agg_inc,
					   dict_income_agg,
					   list_feats_raw_debt,
					   list_feats_agg_debt,
					   dict_debt_agg,
					   list_feats_raw_ln,
					   list_feats_raw_tuaccept,
					   list_feats_raw_cvlink,
					   df_empty,
					   pipeline_pd=None):# se to None for testing
		# args
		self.list_feats_raw_app = list_feats_raw_app
		self.list_feats_raw_inc = list_feats_raw_inc
		self.list_feats_agg_inc = list_feats_agg_inc
		self.dict_income_agg = dict_income_agg
		self.list_feats_raw_debt = list_feats_raw_debt
		self.list_feats_agg_debt = list_feats_agg_debt
		self.dict_debt_agg = dict_debt_agg
		self.list_feats_raw_ln = list_feats_raw_ln
		self.list_feats_raw_tuaccept = list_feats_raw_tuaccept
		self.list_feats_raw_cvlink = list_feats_raw_cvlink
		self.df_empty = df_empty
		self.pipeline_pd = pipeline_pd
	# payload for each applicant
	def get_payload_df(self, json_str_request):
		# get the payload for each applicant
		list_unique_id = []
		list_payload = []
		for applicant in json_str_request['rows']:
			# get unique_id
			unique_id = applicant['row_id']
			list_unique_id.append(unique_id)
			# get data tables
			payload = applicant['sources']
			list_payload.append(payload)
		# create df with payload
		df_payload =  pd.DataFrame({'list_payload': list_payload})
		# save output to self
		self.list_unique_id = list_unique_id
		self.df_payload = df_payload
		# return object
		return self
	# define parse_application
	def parse_application(self, str_values):
		# create error
		self.error_app = ''
		# put into df
		df_app = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col in self.list_feats_raw_app)
		# append __app to each column name except ApplicationDate
		df_app.columns = [f'{col}__app' for col in df_app.columns]
		# if df_app is all na append an error
		if df_app.isnull().all().all():
			self.error_app = 'No application data'
		# reset index
		df_app.reset_index(drop=True, inplace=True)
		# save df_app to self
		self.df_app = df_app
		# return object
		return self
	# define parse_income
	def parse_income(self, str_values):
		# create error
		self.error_inc = ''
		# put into df
		df_inc = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col in self.list_feats_raw_inc)
		# check if df is empty
		if df_inc.empty:
			# create error
			self.error_inc = 'No income data'
			# create empty df with list_feats_inc_agg as cols
			df_inc = pd.DataFrame(columns=self.list_feats_agg_inc)
			# create row with nan
			df_inc = df_inc.append(pd.Series(), ignore_index=True)
		else:
			# filter rows
			df_inc = df_inc[(df_inc['bitInvalid']==False) & (df_inc['bitUse']==True)]
			# if df_inc is empty after filtering on bit cols
			if df_inc.empty:
				# create error
				self.error_inc = 'No income data due to bit filter'
				# create empty df with list_feats_inc_agg as cols
				df_inc = pd.DataFrame(columns=self.list_feats_agg_inc)
				# create row with nan
				df_inc = df_inc.append(pd.Series(), ignore_index=True)
			else:
				# append __income to each column name
				df_inc.columns = [f'{col}__income' for col in df_inc.columns]
				# aggregate
				df_inc = df_inc.groupby('UniqueID__income', as_index=False).agg(self.dict_income_agg)
				# rename columns
				df_inc.columns = [f'{tpl_col[0]}_{tpl_col[1]}' for tpl_col in list(df_inc.columns)]
		# make sure bottom row is selected
		df_inc = pd.DataFrame(df_inc.iloc[-1, :]).T
		# reset index
		df_inc.reset_index(drop=True, inplace=True)
		# save to object
		self.df_inc = df_inc
		# return object
		return self	
	# define parse_debt
	def parse_debt(self, str_values):
		# create error
		self.error_debt = ''
		# put into df
		df_debt = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col in self.list_feats_raw_debt)
		# check if df is empty
		if df_debt.empty:
			# create error
			self.error_debt = 'No debt data'
			# create empty df with list_feats_debt_agg as cols
			df_debt = pd.DataFrame(columns=self.list_feats_agg_debt)
			# create row with nan
			df_debt = df_debt.append(pd.Series(), ignore_index=True)
		else:
			# filter rows
			df_debt = df_debt[(df_debt['bitInvalid']==False) & (df_debt['bitUse']==True)]
			# if df_debt is empty after filtering on bit cols
			if df_debt.empty:
				# create errors
				self.error_debt = 'No debt data due to bit filter'
				# create empty df with list_feats_debt_agg as cols
				df_debt = pd.DataFrame(columns=self.list_feats_agg_debt)
				# create row with nan
				df_debt = df_debt.append(pd.Series(), ignore_index=True)
			else:
				# append __debt to each column name
				df_debt.columns = [f'{col}__debt' for col in df_debt.columns]
				# aggregate
				df_debt = df_debt.groupby('UniqueID__debt', as_index=False).agg(self.dict_debt_agg)
				# rename columns
				df_debt.columns = [f'{tpl_col[0]}_{tpl_col[1]}' for tpl_col in list(df_debt.columns)]
		# make sure bottom row is selected
		df_debt = pd.DataFrame(df_debt.iloc[-1, :]).T
		# reset index
		df_debt.reset_index(drop=True, inplace=True)
		# save to object
		self.df_debt = df_debt
		# return object
		return self
	# define parse_ln
	def parse_ln(self, str_values):
		# create error
		self.error_ln = ''
		# put into df
		df_ln = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col in self.list_feats_raw_ln)
		# append __ln to each column name
		df_ln.columns = [f'{col}__ln' for col in df_ln.columns]
		# check if df is empty
		if df_ln.empty:
			# create error
			self.error_ln = 'No Lexis Nexis data'
			# create row with nan
			df_ln = df_ln.append(pd.Series(), ignore_index=True)
		# make sure bottom row is selected
		df_ln = pd.DataFrame(df_ln.iloc[-1, :]).T
		# reset index
		df_ln.reset_index(drop=True, inplace=True)
		# save to object
		self.df_ln = df_ln
		# return object
		return self
	# define parse_tuxml
	def parse_tuxml(self, str_values):
		# create error
		self.error_tuxml = ''
		# define helper function for parsing XML
		def parse_xml(root, list_feats_raw_tuaccept, list_feats_raw_cvlink):
			# save child table vals so we don't have to write it a bunch
			str_child_table_vals = '{http://www.transunion.com/namespace}value'
			# get col names and values
			list_col_name = []
			list_col_value = []
			# iterate through child branches
			for child in root.iter(tag='{http://www.transunion.com/namespace}characteristic'): # child table
				# get col name
				col_name = child.find('{http://www.transunion.com/namespace}id').text # child table cols
				# logic
				if col_name.upper() in list_feats_raw_cvlink:
					# append __cvlink
					col_name = f'{col_name.upper()}__tucvlink'
					bool_in_payload = True
				elif col_name.lower() in list_feats_raw_cvlink:
					# append __cvlink
					col_name = f'{col_name.upper()}__tucvlink'
					bool_in_payload = True
				elif col_name.upper() in list_feats_raw_tuaccept:
					# append __tuaccept
					col_name = f'{col_name.upper()}__tuaccept'
					bool_in_payload = True
				elif col_name.lower() in list_feats_raw_tuaccept:
					# append __tuaccept
					col_name = f'{col_name.lower()}__tuaccept'
					bool_in_payload = True
				else:
					bool_in_payload = False
				# logic
				if bool_in_payload:
					# append col_name
					list_col_name.append(col_name)
					# try getting the value
					try:
						# get col value
						col_value = child.find(str_child_table_vals).text # child table vals
					# if its None
					except AttributeError:
						# set col value to np.nan
						col_value = np.nan
					# append col_value to list_col_value
					list_col_value.append(col_value)
			# define helper for helper function for converting to proper data type
			def convert_proper_dtype(str_value):
				# try converting to integer
				try:
					col_val = int(str_value)
				# if it cannot be converted to integer
				except:
					# try converting to float
					try:
						col_val = float(str_value)
					# if it cannot be converted to float
					except:
						col_val = str_value
				# return
				return col_val
			# convert each string value in list_col_value to appropriate value
			list_col_value = list(pd.Series(list_col_value).apply(lambda x: convert_proper_dtype(str_value=x)))
			# zip into dictionary
			dict_ = dict(zip(list_col_name, list_col_value))
			# put into df
			return pd.DataFrame(dict_, index=[0])
		# read data from string
		root = ET.fromstring(str_values)
		# parse xml
		df_tuxml = parse_xml(root=root, 
							 list_feats_raw_tuaccept=self.list_feats_raw_tuaccept,
							 list_feats_raw_cvlink=self.list_feats_raw_cvlink)
		# if df_tuxml is empty
		if df_tuxml.empty:
			# create error
			self.error_tu = 'No TU data'
			# create a row with nan
			df_tuxml = df_tuxml.append(pd.Series(), ignore_index=True)
		# make sure bottom row is selected
		df_tuxml = pd.DataFrame(df_tuxml.iloc[-1, :]).T
		# reset index
		df_tuxml.reset_index(drop=True, inplace=True)
		# save to object
		self.df_tuxml = df_tuxml
		# return object
		return self
	# define parse all
	def parse_all(self, json_str_request):
		# get payload df
		self.get_payload_df(json_str_request=json_str_request)
		# empty lists of lists
		list_list_errors = []
		list_list_df = []
		# iterate through payloads
		for list_payload in self.df_payload['list_payload']:
			# empty lists
			list_errors = []
			list_df = []
			# iterate through the tables
			for dict_data in list_payload:
				# get values
				str_values = dict_data['values']
				# application
				if dict_data['name'] == 'Application':
					self.parse_application(str_values=str_values)
					# append
					list_errors.append(self.error_app)
					list_df.append(self.df_app)
				# income
				if dict_data['name'] == 'Incomes':
					# parse income
					self.parse_income(str_values=str_values)
					# append 
					list_errors.append(self.error_inc)
					list_df.append(self.df_inc)
				# debt
				if dict_data['name'] == 'Debts':
					# parse debt
					self.parse_debt(str_values=str_values)
					# append 
					list_errors.append(self.error_debt)
					list_df.append(self.df_debt)
				# ln
				if dict_data['name'] == 'Lexis Nexis Risk View 5':
					# parse ln
					self.parse_ln(str_values=str_values)
					# append
					list_errors.append(self.error_ln)
					list_df.append(self.df_ln)
				# tu
				if dict_data['name'] == 'TUXML':
					# parse xml
					self.parse_tuxml(str_values=str_values)
					# append
					list_errors.append(self.error_tuxml)
					list_df.append(self.df_tuxml)
			# append list_errors to list_list_errors
			list_list_errors.append(list_errors)
			# append list_df to list_list_df
			list_list_df.append(list_df)
		# save lists to object
		self.list_list_errors = list_list_errors
		self.list_list_df = list_list_df
		# return object
		return self
	# define create_x
	def create_x(self, json_str_request):
		# parse all
		self.parse_all(json_str_request=json_str_request)
		# concatenate columns of each df in each list (horizontally i.e., axis=1)
		list_df_concat = []
		for list_dfs in self.list_list_df:
			# concatenate columns (i.e., column bind)
			df_concat = pd.concat(list_dfs, axis=1, sort=False) # WORKING PROPERLY		
			# append to list
			list_df_concat.append(df_concat)
		# concatenate rows (vertically)
		X = pd.concat(list_df_concat, axis=0, sort=False)
		# ensure there is a field for every feature
		X = pd.concat([self.df_empty, X], axis=0, sort=False) # WORKING PROPERLY
		# save to object
		self.X = X
		# return object
		return self
