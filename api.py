# api

# define pipeline class
class PipelineDataPrep:
	# initialize
	def __init__(self, list_transformers):
		self.list_transformers = list_transformers
	# prep data
	def prep_x(self, X):
		# loop through transformers
		for transformer in self.list_transformers:
			# transform
			X = transformer.transform(X)
		# return
		return X

