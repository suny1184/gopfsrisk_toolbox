# api
from sklearn.base import BaseEstimator, TransformerMixin

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
	# prep data
	def prep_x(self, X):
		# loop through transformers
		for transformer in self.list_transformers:
			# transform
			X = transformer.transform(X)
		# save X to object
		self.X = X
		# return
		return self
	# predict
	def get_prediction(self):
		# generate predictions
		flt_prediction = self.model.predict_proba(self.X)[0,1]
		# return
		return flt_prediction
	# combine both methods
	def prep_predict(self, X):
		# call prep_x
		self.prep_x(X)
		# return get_prediction
		return self.get_prediction()


