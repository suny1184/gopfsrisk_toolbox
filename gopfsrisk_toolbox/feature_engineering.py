# feature engineering

# create fe class
class FeatureEngineeringAaron:
	# transform
	def transform(self, X):
		# debt to income
		try:
			X['debt_to_income'] = X['fltBalanceCurrent__debt_sum'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		# day relative to year
		try:
			X['ApplicationDate_day_year_tan'] = X['ApplicationDate_day_year_sin'] / X['ApplicationDate_day_year_cos']
		except:
			pass
		# day relative to month
		try:
			X['ApplicationDate_day_month_tan'] = X['ApplicationDate_day_month_sin'] / X['ApplicationDate_day_month_cos']
		except:
			pass
		# day relative to week
		try:
			X['ApplicationDate_day_week_tan'] = X['ApplicationDate_day_week_sin'] / X['ApplicationDate_day_week_cos']
		except:
			pass
		# month relative to year
		try:
			X['ApplicationDate_month_year_tan'] = X['ApplicationDate_month_year_sin'] / X['ApplicationDate_month_year_cos']
		except:
			pass
		# hour relative to day
		try:
			X['ApplicationDate_hour_day_tan'] = X['ApplicationDate_hour_day_sin'] / X['ApplicationDate_hour_day_cos']
		except:
			pass
		# return
		return X

"""
class FeatureEngineeringAndrew:
    def __init__(self, bool_drop_feats=True):
        self.bool_drop_feats = bool_drop_feats
    # transform
    def transform(self,X):
        # removing pseudo-missing values (negatives) with NaN, collecting a mean, then imputing any completely missing with 0
        # deleting original columns afterwards
        # this takes a long time to complete
        try:
            X['balances'] = X.loc[:,'agg101__tuaccept':'agg124__tuaccept'].where(X.loc[:,'agg101__tuaccept':'agg124__tuaccept']>0,np.nan).apply(lambda x: np.nanmean(x), axis=1)
            X['balances'] = X['balances'].fillna(0)
            # here, create a list of features to drop
            if self.bool_drop_feats:
                for col in X.loc[:,'agg101__tuaccept':'agg124__tuaccept'].columns:
                    if col in list(X.columns):
                        del X[col]
        except:
            pass
        try:
            X['credit_line'] = X.loc[:,'agg201__tuaccept':'agg224__tuaccept'].where(X.loc[:,'agg201__tuaccept':'agg224__tuaccept']>0,np.nan).apply(lambda x: np.nanmean(x), axis=1)
            X['credit_line'] = X['credit_line'].fillna(0)
            # here, create a list of features to drop
            if self.bool_drop_feats:
                for col in X.loc[:,'agg201__tuaccept':'agg224__tuaccept'].columns:
                    if col in list(X.columns):
                        del X[col]
        except:
            pass
        try:
            X['amount_past_due'] = X.loc[:,'agg301__tuaccept':'agg324__tuaccept'].where(X.loc[:,'agg301__tuaccept':'agg324__tuaccept']>0,np.nan).apply(lambda x: np.nanmean(x), axis=1)
            X['amount_past_due'] = X['amount_past_due'].fillna(0)
            # here, create a list of features to drop
            if self.bool_drop_feats:
                for col in X.loc[:,'agg301__tuaccept':'agg324__tuaccept'].columns:
                    if col in list(X.columns):
                        del X[col]
        except:
            pass
        try:
            X['agg_spending'] = X.loc[:,'aggs101__tuaccept':'aggs124__tuaccept'].where(X.loc[:,'aggs101__tuaccept':'aggs124__tuaccept']>0,np.nan).apply(lambda x: np.nanmean(x), axis=1)
            X['balances'] = X['balances'].fillna(0)
            # here, create a list of features to drop
            if self.bool_drop_feats:
                for col in X.loc[:,'aggs101__tuaccept':'aggs124__tuaccept'].columns:
                    if col in list(X.columns):
                        del X[col]
        except:
            pass
        return X
"""