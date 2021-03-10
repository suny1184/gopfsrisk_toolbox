# feature engineering (for API)

# create fe class
class FeatureEngineeringAaron:
    # transform
    def transform(X):
    	try:
        	X['debt_to_income'] = X['fltBalanceCurrent__debt_sum'] / X['fltGrossMonthly__income_sum']
        except:
        	pass
        # return
        return X
