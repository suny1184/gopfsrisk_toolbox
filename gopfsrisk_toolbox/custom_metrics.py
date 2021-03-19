# custom metrics
from sklearn.metrics import average_precision_score
import numpy as np
from scipy.special import expit

# define class for precision-recall AUC (catboost)
class PrecisionRecallAUC:
	# define a static method to use in evaluate method
	@staticmethod
	def get_pr_auc(y_true, y_pred):
		# fit predictions to logistic sigmoid function
		y_pred = expit(y_pred).astype(float)
		# actual values should be 1 or 0 integers
		y_true = y_true.astype(int)
		# calculate average precision
		return average_precision_score(y_true=y_true, y_score=y_pred)	
	# Returns whether great values of metric error are better
	def is_max_optimal(self):
		return True
	# Compute metric
	def evaluate(self, approxes, target, weight):
		# make sure theres only 1 item in approxes
		assert len(approxes) == 1
		# make sure there are as many actual (target) as there are predictions (approxes[0])
		assert len(target) == len(approxes[0])
		# set target to integer and save as y_true
		y_true = np.array(target).astype(int)
		# get predictions
		y_pred = approxes[0]
		# generate score (we will call it error just to make things work)
		error = self.get_pr_auc(y_true=y_true, y_pred=y_pred)
		# generate weight
		if not weight: # i.e. if list_class_weights is None
			weight = 1
		else:
			weight = weight[1]
		# return
		return error, weight
	# get final error   
	def get_final_error(self, error, weight):
		# Returns final value of metric based on error and weight
		return error * weight