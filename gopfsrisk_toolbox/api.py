# api
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
	def __init__(self, list_transformers, model):
		self.list_transformers = list_transformers
		self.model = model
	# prep predict
	def prep_predict(self, X):
		# loop through transformers
		for transformer in self.list_transformers:
			# transform
			X = transformer.transform(X)
		# make predictions
		y_hat = self.model.predict_proba(X[self.model.feature_names_])[:,1]
		# get mean
		y_hat = np.mean(y_hat)
		# return
		return y_hat