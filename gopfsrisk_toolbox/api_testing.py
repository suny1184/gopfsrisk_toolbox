import ast
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# define class
class TimeParsing:
	# initialize
	def __init__(self, cls_parse_payload, ser_payloads):
		self.cls_parse_payload = cls_parse_payload
		self.ser_payloads = ser_payloads
	# parse
	def parse_payloads(self):
		# list of time
		list_flt_sec = []
		# list of n debtors
		list_int_n_debtors = []
		# iterate 
		for a, json_str_request in enumerate(self.ser_payloads):
			# print message
			print(f'Payload {a+1}/{len(self.ser_payloads)}')
			# convert to dictionary
			json_str_request = ast.literal_eval(json_str_request)
			# start timer
			time_start = time.perf_counter()
			# generate output
			self.cls_parse_payload.generate_output(json_str_request=json_str_request)
			# get time in sec
			flt_sec = time.perf_counter() - time_start
			# append to list_flt_sec
			list_flt_sec.append(flt_sec)
			# get int_n_debtors
			int_n_debtors = len(self.cls_parse_payload.list_unique_id)
			# append to list_int_n_debtors
			list_int_n_debtors.append(int_n_debtors)
			# print current mean time
			print(f'Mean parsing time: {np.mean(list_flt_sec):0.5}')
		# create df
		df_output = pd.DataFrame({'sec': list_flt_sec,
			                      'n_debtors': list_int_n_debtors})
		# save to object
		self.df_output = df_output
		# return
		return self
	# plot
	def create_plot(self, tpl_figsize):
		# ax
		fig, ax = plt.subplots(nrows=3, ncols=1, figsize=tpl_figsize)
		# altogether
		ax[0].set_title(f"All Debtors (mean = {np.mean(self.df_output['sec']):5} sec)")
		sns.histplot(data=self.df_output['sec'], kde=True, ax=[0])
		# 1 debtor
		df_output_1 = self.df_output[self.def_output['n_debtors']==1]
		ax[1].set_title(f"One Debtor (mean = {df_output_1['sec']} sec)")
		sns.histplot(data=df_output_1['sec'], kde=True, ax=[1])
		# 2 debtors
		df_output_2 = self.df_output[self.def_output['n_debtors']==2]
		ax[2].set_title(f"Two Debtors (mean = {df_output_2['sec']} sec)")
		sns.histplot(data=df_output_2['sec'], kde=True, ax=[2])
		# save to object
		self.df_output_1 = df_output_1
		self.df_output_2 = df_output_2
		self.fig = fig
		# return
		return self




