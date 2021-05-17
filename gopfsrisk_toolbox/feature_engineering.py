# feature engineering
import numpy as np
import time

# create fe class
class FeatureEngineeringAaronPDLGDLower:
	# transform
	def transform(self, X):
		time_start = time.perf_counter()
		# from James
		# down payment to amount financed
		try:
			X['eng_down_to_financed'] = X['fltapproveddowntotal__app'] / X['fltamountfinanced__app']
		except:
			pass
		# down payment over gross monthly
		try:
			X['eng_down_to_income'] = X['fltapproveddowntotal__app'] / X['fltgrossmonthly__income_sum']
		except:
			pass
		# down payment to price wholesale
		try:
			X['eng_down_to_wholesale'] = X['fltapproveddowntotal__app'] / X['fltapprovedpricewholesale__app']
		except:
			pass
		# Cyclic: Month relative to year
		# sin
		try:
			X['eng_applicationmonth__app_sin'] = np.sin((X['applicationmonth__app']-1) * (2*np.pi/12))
		except:
			pass
		# cos
		try:
			X['eng_applicationmonth__app_cos'] = np.cos((X['applicationmonth__app']-1) * (2*np.pi/12))
		except:
			pass
		# tan
		try:
			X['eng_applicationmonth__app_tan'] = X['eng_applicationmonth__app_sin'] / X['eng_applicationmonth__app_cos']
		except:
			pass
		# Cyclic: Quarter relative to year
		# sin
		try:
			X['eng_applicationquarter__app_sin'] = np.sin((X['applicationquarter__app']-1) * (2*np.pi/4))
		except:
			pass
		# cos
		try:
			X['eng_applicationquarter__app_cos'] = np.cos((X['applicationquarter__app']-1) * (2*np.pi/4))
		except:
			pass
		# tan
		try:
			X['eng_applicationquarter__app_tan'] = X['eng_applicationquarter__app_sin'] / X['eng_applicationquarter__app_cos']
		except:
			pass
		# loan to value
		try:
			X['eng_loan_to_value'] = X['fltamountfinanced__app'] / X['fltapprovedpricewholesale__app']
		except:
			pass
		# debt to income
		try:
			X['eng_debt_to_income'] = X['fltmonthlypayment__debt_mean'] / X['fltgrossmonthly__income_sum']
		except:
			pass
		print(f'Time to feature engineer: {(time.perf_counter()-time_start):0.5} sec.')
		# return
		return X

# create fe class
class FeatureEngineeringAaronPD:
	# transform
	def transform(self, X):
		time_start = time.perf_counter()
		# from James
		# down payment to amount financed
		try:
			X['ENG_down_to_financed'] = X['fltApprovedDownTotal__app'] / X['fltAmountFinanced__app']
		except:
			pass
		# down payment over gross monthly
		try:
			X['ENG_down_to_income'] = X['fltApprovedDownTotal__app'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		# down payment to price wholesale
		try:
			X['ENG_down_to_wholesale'] = X['fltApprovedDownTotal__app'] / X['fltApprovedPriceWholesale__app']
		except:
			pass
		# Cyclic: Month relative to year
		# sin
		try:
			X['ENG_ApplicationMonth__app_sin'] = np.sin((X['ApplicationMonth__app']-1) * (2*np.pi/12))
		except:
			pass
		# cos
		try:
			X['ENG_ApplicationMonth__app_cos'] = np.cos((X['ApplicationMonth__app']-1) * (2*np.pi/12))
		except:
			pass
		# tan
		try:
			X['ENG_ApplicationMonth__app_tan'] = X['ENG_ApplicationMonth__app_sin'] / X['ENG_ApplicationMonth__app_cos']
		except:
			pass
		# Cyclic: Quarter relative to year
		# sin
		try:
			X['ENG_ApplicationQuarter__app_sin'] = np.sin((X['ApplicationQuarter__app']-1) * (2*np.pi/4))
		except:
			pass
		# cos
		try:
			X['ENG_ApplicationQuarter__app_cos'] = np.cos((X['ApplicationQuarter__app']-1) * (2*np.pi/4))
		except:
			pass
		# tan
		try:
			X['ENG_ApplicationQuarter__app_tan'] = X['ENG_ApplicationQuarter__app_sin'] / X['ENG_ApplicationQuarter__app_cos']
		except:
			pass
		# loan to value
		try:
			X['ENG_loan_to_value'] = X['fltAmountFinanced__app'] / X['fltApprovedPriceWholesale__app']
		except:
			pass
		# debt to income
		try:
			X['ENG_debt_to_income'] = X['fltMonthlyPayment__debt_mean'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		print(f'Time to feature engineer: {(time.perf_counter()-time_start):0.5} sec.')
		# return
		return X

# create fe class
class FeatureEngineeringAaronLGD:
	# transform
	def transform(self, X):
		time_start = time.perf_counter()
		# from James
		# down payment to amount financed
		try:
			X['ENG_down_to_financed'] = X['fltApprovedDownTotal__app'] / X['fltAmountFinanced__app']
		except:
			pass
		# down payment over gross monthly
		try:
			X['ENG_down_to_income'] = X['fltApprovedDownTotal__app'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		# down payment to price wholesale
		try:
			X['ENG_down_to_wholesale'] = X['fltApprovedDownTotal__app'] / X['fltApprovedPriceWholesale__app']
		except:
			pass
		# Cyclic: Month relative to year
		# sin
		try:
			X['ENG_ApplicationMonth__app_sin'] = np.sin((X['ApplicationMonth__app']-1) * (2*np.pi/12))
		except:
			pass
		# cos
		try:
			X['ENG_ApplicationMonth__app_cos'] = np.cos((X['ApplicationMonth__app']-1) * (2*np.pi/12))
		except:
			pass
		# tan
		try:
			X['ENG_ApplicationMonth__app_tan'] = X['ENG_ApplicationMonth__app_sin'] / X['ENG_ApplicationMonth__app_cos']
		except:
			pass
		# Cyclic: Quarter relative to year
		# sin
		try:
			X['ENG_ApplicationQuarter__app_sin'] = np.sin((X['ApplicationQuarter__app']-1) * (2*np.pi/4))
		except:
			pass
		# cos
		try:
			X['ENG_ApplicationQuarter__app_cos'] = np.cos((X['ApplicationQuarter__app']-1) * (2*np.pi/4))
		except:
			pass
		# tan
		try:
			X['ENG_ApplicationQuarter__app_tan'] = X['ENG_ApplicationQuarter__app_sin'] / X['ENG_ApplicationQuarter__app_cos']
		except:
			pass
		# loan to value
		try:
			X['ENG_loan_to_value'] = X['fltAmountFinanced__app'] / X['fltApprovedPriceWholesale__app']
		except:
			pass
		# debt to income
		try:
			X['ENG_debt_to_income'] = X['fltMonthlyPayment__debt_mean'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		print(f'Time to feature engineer: {(time.perf_counter()-time_start):0.5} sec.')
		# return
		return X


class FeatureEngineeringJQ:
	# transform
	def transform(self, X):
		# loan to value
		try:
			X['eng_loan_to_value'] = X['fltAmountFinanced__app'] / X['fltApprovedPriceWholesale__app']
		except:
			pass
		# debt to income
# 		try:
# 			X['eng_debt_to_income'] = X['fltMonthlyPayment__debt_mean'] / X['fltGrossMonthly__income_sum']
# 		except:
# 			pass
		# down payment to financed
		try:
			X['eng_down_to_financed'] = X['fltApprovedDownTotal__app'] / X['fltAmountFinanced__app']
		except:
			pass
		# down pmt over grossmonthly
		try:
			X['eng_down_to_income'] = X['fltApprovedDownTotal__app'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		# down pmt over price wholesale
		try:
			X['eng_down_to_wholesale'] = X['fltApprovedDownTotal__app'] / X['fltApprovedPriceWholesale__app']
		except:
			pass

		#TRADE INFO
		# number of open trades / number of trades
		try:
			X['eng_at02s_to_at01s'] = X['at02s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#satisfactory open trades / number of trade
		try:
 			X['eng_at03s_to_at01s'] = X['at03s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#24 months open trade / number of trades
		try:
 			X['eng_at09s_to_at01s'] = X['at09s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#open satisfactory 24 months / number of trades
		try:
 			X['eng_at27s_to_at01s'] = X['at27s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#total past due amount of open trades / total balance of all trades in 12 months
		try:	
			X['eng_at57s_to_at01s'] = X['at57s__tuaccept'] / X['at101s__tuaccept']
		except:
			pass
		
		#AUTO TRADE
		# open auto vs number of auto trades
		try:
			X['eng_au02s_to_au01s'] = X['au02s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# satisfactory auto trades over number auto trades
		try:
			X['eng_au03s_to_au01s'] = X['au03s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# auto trades opened in 24 / number of auto trades
		try:
			X['eng_au09s_to_au01s'] = X['au09s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# months since recent / months since oldest auto trade opened
		try:
			X['eng_au21s_to_au20s'] = X['au21s__tuaccept'] / X['au20s__tuaccept']
		except:
			pass
		# open and satisf auto trades 24 months / number of auto trades
		try:
			X['eng_au27s_to_au01s'] = X['au27s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# open and satisf auto trades 24 months / number of open auto trades
		try:
			X['eng_au27s_to_au02s'] = X['au27s__tuaccept'] / X['au02s__tuaccept']
		except:
			pass
		# open and satisf auto trades 24 months / number of open satisf trades
		try:
			X['eng_au27s_to_au03s'] = X['au27s__tuaccept'] / X['au03s__tuaccept']
		except:
			pass
		
		#CREDIT CARD TRADES
		# open CC trades vs CC trades
		try:
			X['eng_bc02s_to_bc01s'] = X['bc02s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		# current open satisf CC trades vs CC trades
		try:
			X['eng_bc03s_to_bc01s'] = X['bc03s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		# open CC trades 24m vs CC trades
		try:
			X['eng_bc09s_to_bc01s'] = X['bc09s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		# months since most recent vs month since oldest CC 
		try:	
			X['eng_bc21s_to_bc20s'] = X['bc21s__tuaccept'] / X['bc20s__tuaccept']
		except:
			pass
		# open satisf CC trade 24 months vs CC trades
		try:	
			X['eng_bc27s_to_bc01s'] = X['bc27s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		
		#BANK INSTALLMENTS
		# open bank installment vs bank installment trades
		try:	
			X['eng_bi02s_to_bi01s'] = X['bi02s__tuaccept'] / X['bi01s__tuaccept']
		except:
			pass
		# open satisf bank installment vs number bank installment
		try:	
			X['eng_bi12s_to_bi01s'] = X['bi12s__tuaccept'] / X['bi01s__tuaccept']
		except:
			pass
		# months since most recent bi vs months since oldest bi
		try:	
			X['eng_bi21s_to_bi20s'] = X['bi21s__tuaccept'] / X['bi20s__tuaccept']
		except:
			pass
		# utilization open bi verified 12m vs total open bi verified 12m
		try:
			X['eng_bi34s_to_bi33s'] = X['bi34s__tuaccept'] / X['bi33s__tuaccept']
		except:
			pass
		
		#BANK REVOLVER
		# open br trades vs br trades
		try:	
			X['eng_br02s_to_br01s'] = X['br02s__tuaccept'] / X['br01s__tuaccept']
		except:
			pass
		# open satisf br trades vs br trades
		try:
			X['eng_br03s_to_br01s'] = X['br03s__tuaccept'] / X['br01s__tuaccept']
		except:
			pass
		# br trades opened past 24 / br trades opened
		try:
			X['eng_br09s_to_br01s'] = X['br09s__tuaccept'] / X['br01s__tuaccept']
		except:
			pass
		# months recent br / months oldest br
		try:
			X['eng_br21s_to_br20s'] = X['br21s__tuaccept'] / X['br20s__tuaccept']
		except:
			pass
		# open satis 24m / br trades
		try:
			X['eng_br27s_to_br20s'] = X['br27s__tuaccept'] / X['br20s__tuaccept']
		except:
			pass
		
		#CHARGE OFF TRADES
		# number CO trades in past 24 months / CO trades
		try:
			X['eng_co03s_to_co01s'] = X['co03s__tuaccept'] / X['co01s__tuaccept']
		except:
			pass
		# balance CO 24m / CO balance
		try:
			X['eng_co07s_to_co05s'] = X['co07s__tuaccept'] / X['co05s__tuaccept']
		except:
			pass
		
		# FORECLOSURE TRADES
		# foreclosure trades past 24m / foreclosure trades
		try:
			X['eng_fc03s_to_fc01s'] = X['fc03s__tuaccept'] / X['fc01s__tuaccept']
		except:
			pass
		# balance FC trades 24m / balance FC trades
		try:
			X['eng_fc07s_to_fc05s'] = X['fc07s__tuaccept'] / X['fc05s__tuaccept']
		except:
			pass
		
		#FINANCE INSTALLMENT
		# open fi trades / fi trades
		try:
			X['eng_fi02s_to_fi01s'] = X['fi02s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		# open satisf fi trades / fi trades
		try:
			X['eng_fi03s_to_fi01s'] = X['fi03s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		# number fi opened in past 24 months / opened fi trades
		try:	
			X['eng_fi09s_to_fi02s'] = X['fi09s__tuaccept'] / X['fi02s__tuaccept']
		except:
			pass
		# number fi opened past 24m / number fi trades
		try:
			X['eng_fi09s_to_fi01s'] = X['fi09s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		# months most recent fi opened / months since oldest fi opened
		try:
			X['eng_fi21s_to_fi20s'] = X['fi21s__tuaccept'] / X['fi20s__tuaccept']
		except:
			pass
		# number current open satisf fi 24m / number fi trades
		try:
			X['eng_fi27s_to_fi01s'] = X['fi27s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		
		#FINANCE REVOLVING TRADES
		# number of open FR trades / number FR trades
		try:
			X['eng_fr02s_to_fr01s'] = X['fr02s__tuaccept'] / X['fr01s__tuaccept']
		except:
			pass
		# number current satisf open fr / number FR trades
		try:
			X['eng_fr03s_to_fr01s'] = X['fr03s__tuaccept'] / X['fr01s__tuaccept']
		except:
			pass
		# number opened fr trades 24m / number FR trades
		try:
			X['eng_fr09s_to_fr01s'] = X['fr09s__tuaccept'] / X['fr01s__tuaccept']
		except:
			pass
		
		# HOME EQUITY
		# open home equity vs number of home equity loans
		try:
			X['eng_hi02s_to_hi01s'] = X['hi02s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		# current satisf open he vs number he loans
		try:
			X['eng_hi03s_to_hi01s'] = X['hi03s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		# number he opened past 24m / number he loans
		try:
			X['eng_hi09s_to_hi01s'] = X['hi09s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		# months since most recent he opened / months since oldest
		try:
			X['eng_hi21s_to_hi20s'] = X['hi21s__tuaccept'] / X['hi20s__tuaccept']
		except:
			pass
		# number currently open satisf he loan 24m / number he loans
		try:
			X['eng_hi27s_to_hi01s'] = X['hi27s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		
		# HOME EQUITY LOC
		# number he open LOC / number he LOC
		try:
			X['eng_hr02s_to_hr01s'] = X['hr02s__tuaccept'] / X['hr01s__tuaccept']
		except:
			pass
		# number he opened LOC 24m / number he LOC
		try:
			X['eng_hr12s_to_hr01s'] = X['hr12s__tuaccept'] / X['hr01s__tuaccept']
		except:
			pass
		# months since most recent opened vs months oldest
		try:
			X['eng_hr21s_to_hr20s'] = X['hr21s__tuaccept'] / X['hr20s__tuaccept']
		except:
			pass
		
		# INSTALLMENT TRADES
		# number open installments vs installment trades
		try:
			X['eng_in02s_to_in01s'] = X['in02s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# current open satisf vs installment trades
		try:
			X['eng_in03s_to_in01s'] = X['in03s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# number opened past 24m vs installment trades
		try:
			X['eng_in09s_to_in01s'] = X['in09s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# number open verified in past 12 months vs installment trades
		try:
			X['eng_in12s_to_in01s'] = X['in12s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# months since most recent vs months oldest
		try:
			X['eng_in21s_to_in20s'] = X['in21s__tuaccept'] / X['in20s__tuaccept']
		except:
			pass
		# open satisf 24m vs installment trades
		try:
			X['eng_in27s_to_in01s'] = X['in27s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# open verified 12m vs installment trades
		try:
			X['eng_in28s_to_in01s'] = X['in28s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		
		# LOAN MODIFICATIONS
		# number LM mortage 90+DPD vs LM mortgage
		try:
			X['eng_lm08s_to_lm01s'] = X['lm08s__tuaccept'] / X['lm01s__tuaccept']
		except:
			pass
		# bank backed LM vs LM 
		try:
			X['eng_lm25s_to_lm01s'] = X['lm25s__tuaccept'] / X['lm01s__tuaccept']
		except:
			pass
		
		# MORTGAGE TRADES
		# number of open mortgage trades vs number or mortgage trades
		try:
			X['eng_mt02s_to_mt01s'] = X['mt02s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		#number of current satisf MT vs mortgage trades
		try:
			X['eng_mt03s_to_mt01s'] = X['mt03s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		# mt trades opened in 24 months vs mt trades
		try:
			X['eng_mt09s_to_mt01s'] = X['mt09s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		# open verified in past 12 months vs mortgage trades
		try:
			X['eng_mt12s_to_mt01s'] = X['mt12s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		#months most recent opened vs oldest opened
		try:
			X['eng_mt21s_to_mt20s'] = X['mt21s__tuaccept'] / X['mt20s__tuaccept']
		except:
			pass
		# number open satisf MT 24 months vs MT
		try:
			X['eng_mt27s_to_mt01s'] = X['mt27s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		
		# joint trade info
		# trades to joint trades open/satisf
		try:
			X['eng_at03s_to_jt03s'] = X['at03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass 
		# auto trades to joint trades open/satisf
		try:
			X['eng_au03s_to_jt03s'] = X['au03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# credit card to joint trades open/satisf
		try:
			X['eng_bc03s_to_jt03s'] = X['bc03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# bank installments vs JT open/satisf
		try:
			X['eng_bi03s_to_jt03s'] = X['bi03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass 
		# bank revolver vs JT open/satisf
		try:
			X['eng_br03s_to_jt03s'] = X['br03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# charge offs in 24 months vs JT open/satisf
		try:
			X['eng_co03s_to_jt03s'] = X['co03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# foreclosure in 24 months vs JT open/satisf
		try:
			X['eng_fc03s_to_jt03s'] = X['fc03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# finance installments vs JT open/satisf
		try:
			X['eng_fi03s_to_jt03s'] = X['fi03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# home equity v JT open/satisf
		try:
			X['eng_hi03s_to_jt03s'] = X['hi03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# home equity LOC v JT open/satisf
		try:
			X['eng_hr03s_to_jt03s'] = X['hr03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# installment trades v JT open/satisf
		try:
			X['eng_in03s_to_jt03s'] = X['in03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# mortgage trades vs JT open/satisf
		try:
			X['eng_mt03s_to_jt03s'] = X['mt03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		
		# joint trades vs individual trades, trades vs application data, bankruptcies, income over trade/ trade over income
		# joint trades
		#-- DONE -- X['jt01s__tuaccept'] total joint trades  test against auto, credit card, bi, br, CO, Foreclosure, home equity, installments
		#-- DONE -- X['jt03s__tuaccept'] number open + satisfactory
		# X['jt21s__tuaccept'] / X['jt20s__tuaccept'] months most recent / months oldest
		
		#income and tradeline info
		# 32s 33s 34s 35s max, total, utilization, average verified 12 months vs income
		# ================================== trades ==============================================
		try:
			X['eng_at32s_to_income'] = X['at32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_at33a_to_income'] = X['at33a__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_at33b_to_income'] = X['at33b__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_at34a_to_income'] = X['at34a__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_at34b_to_income'] = X['at34b__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_at35a_to_income'] = X['at35a__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_at35b_to_income'] = X['at35b__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#================================= auto trades ========================================
		try:
			X['eng_au32s_to_income'] = X['au32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_au33s_to_income'] = X['au33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_au34s_to_income'] = X['au34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_au35s_to_income'] = X['au35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#=============================== credit cards =========================================
		try:
			X['eng_bc32s_to_income'] = X['bc32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_bc33s_to_income'] = X['bc33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_bc34s_to_income'] = X['bc34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_bc35s_to_income'] = X['bc35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#=============================== bank installments ====================================
		try:
			X['eng_bi32s_to_income'] = X['bi32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_bi33s_to_income'] = X['bi33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_bi34s_to_income'] = X['bi34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_bi35s_to_income'] = X['bi35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#============================= bank revolvers ========================================
		try:
			X['eng_br32s_to_income'] = X['br32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_br33s_to_income'] = X['br33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_br34s_to_income'] = X['br34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_br35s_to_income'] = X['br35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#============================ finance installments ===================================
		try:
			X['eng_fi32s_to_income'] = X['fi32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_fi33s_to_income'] = X['fi33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_fi34s_to_income'] = X['fi34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_fi35s_to_income'] = X['fi35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#============================ finance revolvers ======================================
		try:
			X['eng_fr32s_to_income'] = X['fr32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_fr33s_to_income'] = X['fr33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_fr34s_to_income'] = X['fr34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_fr35s_to_income'] = X['fr35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#============================ home equity ============================================
		try:
			X['eng_hi32s_to_income'] = X['hi32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_hi33s_to_income'] = X['hi33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_hi34s_to_income'] = X['hi34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_hi35s_to_income'] = X['hi35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#========================== HE LOC ==================================================
		try:
			X['eng_hr32s_to_income'] = X['hr32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
# 		try:
# 			X['eng_hr33s_to_income'] = X['hr33s__tuaccept'] / X['fltGrossMonthly__income_sum']
# 		except:
# 			pass
		try:
			X['eng_hr34s_to_income'] = X['hr34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
# 		try:
# 			X['eng_hr35s_to_income'] = X['hr35s__tuaccept'] / X['fltGrossMonthly__income_sum']
# 		except:
# 			pass
		
		#============================ installment trades =====================================
		try:
			X['eng_in32s_to_income'] = X['in32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_in33s_to_income'] = X['in33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_in34s_to_income'] = X['in34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_in35s_to_income'] = X['in35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		
		#=========================== mortgage trades =========================================
		try:
			X['eng_mt32s_to_income'] = X['mt32s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_mt33s_to_income'] = X['mt33s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_mt34s_to_income'] = X['mt34s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass
		try:
			X['eng_mt35s_to_income'] = X['mt35s__tuaccept'] / X['fltGrossMonthly__income_sum']
		except:
			pass

		# BK fields
		# X['g094s__tuaccept'] number of public record bankruptcies
		# X['g099s__tuaccept'] number of public BK past 24 months
		# X['g100s__tuaccept'] number of tradeline BK
				
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