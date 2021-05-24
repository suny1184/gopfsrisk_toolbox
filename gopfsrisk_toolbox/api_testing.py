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
		# list of parsed dfs
		list_x_parsed = []
		# list of dictionaries missing
		list_dict_n_miss = []
		# list of errors
		list_list_errors_flat = []
		# list of adverse action reasons
		list_list_reasons_flat = []
		# list of time to get payloads
		list_flt_sec_get_payloads = []
		# list of time to parse
		list_flt_sec_parse = []
		# list of time to create x
		list_flt_sec_create_x = []
		# list of time to preprocess
		list_flt_sec_preprocessing = []
		# list of time to predict
		list_flt_sec_predict = []
		# list of time to get adverse action
		list_flt_sec_adv_act = []
		# list of time to get output
		list_flt_sec_gen_output = []
		# list of time (overall)
		list_flt_sec = []
		# list of n debtors
		list_int_n_debtors = []
		# list of error indices
		list_idx_errors = []
		# iterate 
		for a, json_str_request in enumerate(self.ser_payloads):
			# print message
			print(f'Payload {a+1}/{len(self.ser_payloads)}')
			# convert to dictionary
			try:
				json_str_request = ast.literal_eval(json_str_request)
				# start timer
				time_start = time.perf_counter()
				# generate output
				self.cls_parse_payload.generate_output(json_str_request=json_str_request)
				# get time in sec
				flt_sec = time.perf_counter() - time_start
				# append to list_flt_sec
				list_flt_sec.append(flt_sec)
				# flatten
				list_errors_flat = [item for sublist in self.cls_parse_payload.list_list_errors for item in sublist]
				# flatten
				list_reasons_flat = [item for sublist in self.cls_parse_payload.list_list_reasons for item in sublist]
				# append to lists
				list_x_parsed.append(self.cls_parse_payload.X)
				list_dict_n_miss.append(self.cls_parse_payload.dict_n_miss)
				list_list_errors_flat.append(list_errors_flat)
				list_list_reasons_flat.append(list_reasons_flat)
				list_flt_sec_get_payloads.append(self.cls_parse_payload.flt_sec_get_payloads)
				list_flt_sec_parse.append(self.cls_parse_payload.flt_sec_parse)
				list_flt_sec_create_x.append(self.cls_parse_payload.flt_sec_create_x)
				list_flt_sec_preprocessing.append(self.cls_parse_payload.flt_sec_preprocessing)
				list_flt_sec_predict.append(self.cls_parse_payload.flt_sec_predict)
				list_flt_sec_adv_act.append(self.cls_parse_payload.flt_sec_adv_act)
				list_flt_sec_gen_output.append(self.cls_parse_payload.flt_sec_gen_output)
				# get int_n_debtors
				int_n_debtors = len(self.cls_parse_payload.list_unique_id)
				# append to list_int_n_debtors
				list_int_n_debtors.append(int_n_debtors)
				# print current mean time
				print(f'Mean parsing time: {np.mean(list_flt_sec):0.5}')
			except ValueError: # malformed node or string
				list_idx_errors.append(a)
		# create df
		df_output = pd.DataFrame({'x_parsed': list_x_parsed,
								  'n_miss': list_dict_n_miss,
								  'errors': list_list_errors_flat,
								  'reasons': list_list_reasons_flat,
								  'sec': list_flt_sec,
								  'sec_get_payloads': list_flt_sec_get_payloads,
								  'sec_parse': list_flt_sec_parse,
								  'sec_create_x': list_flt_sec_create_x,
								  'sec_preprocessing': list_flt_sec_preprocessing,
								  'sec_predict': list_flt_sec_predict,
								  'sec_adv_act': list_flt_sec_adv_act,
								  'sec_gen_output': list_flt_sec_gen_output,
			                      'n_debtors': list_int_n_debtors})
		# save to object
		self.list_idx_errors = list_idx_errors
		self.df_output = df_output
		# return
		return self
	# plot
	def create_plot(self, tpl_figsize, str_filename):
		# ax
		fig, ax = plt.subplots(nrows=5, ncols=1, figsize=tpl_figsize)
		# altogether
		flt_mean_all = np.mean(self.df_output['sec'])
		ax[0].set_title(f"All Debtors (mean = {flt_mean_all:0.5} sec; N = {self.df_output.shape[0]})")
		sns.distplot(self.df_output['sec'], kde=True, ax=ax[0])
		# 1 debtor
		df_output_1 = self.df_output[self.df_output['n_debtors']==1]
		flt_mean_1 = np.mean(df_output_1['sec'])
		ax[1].set_title(f"One Debtor (mean = {flt_mean_1:0.5} sec; N = {df_output_1.shape[0]})")
		sns.distplot(df_output_1['sec'], kde=True, ax=ax[1])
		# 2 debtors
		df_output_2 = self.df_output[self.df_output['n_debtors']==2]
		flt_mean_2 = np.mean(df_output_2['sec'])
		ax[2].set_title(f"Two Debtors (mean = {flt_mean_2:0.5} sec; N = {df_output_2.shape[0]})")
		sns.distplot(df_output_2['sec'], kde=True, ax=ax[2])
		# bar plot
		ax[3].set_title('Mean Seconds by N Debtors')
		ax[3].bar(['All', '1 Debtor', '2 Debtors'], [flt_mean_all, flt_mean_1, flt_mean_2])
		# create df for grouped bar plot
		list_int_n_debtors = []
		list_str_step_name = []
		list_flt_sec_mean = []
		for int_n_debtors in [1,2]:
			for str_step_name in ['sec_get_payloads','sec_parse','sec_create_x','sec_preprocessing','sec_predict','sec_adv_act','sec_gen_output']:
				# subset
				df_tmp = self.df_output[self.df_output['n_debtors']==int_n_debtors]
				# get mean of step
				flt_sec_mean = np.mean(df_tmp[str_step_name])
				# append to lists
				list_int_n_debtors.append(int_n_debtors)
				list_str_step_name.append(str_step_name)
				list_flt_sec_mean.append(flt_sec_mean)
		# make df
		df_for_plot = pd.DataFrame({'N Debtors': list_int_n_debtors,
			                        'Steps': list_str_step_name,
			                        'Mean Seconds': list_flt_sec_mean})
		# dictionary for mapping steps
		dict_map_steps = {'sec_get_payloads': 'Get Payloads',
						  'sec_parse': 'Parse',
						  'sec_create_x': 'Create X',
						  'sec_preprocessing': 'Preprocessing',
						  'sec_predict': 'Predict',
						  'sec_adv_act': 'Adverse Action',
						  'sec_gen_output': 'Generate Output'}
		# map
		df_for_plot['Steps'] = df_for_plots['Steps'].map(dict_map_steps)
		# title
		ax[4].set_title('Mean Seconds by Parsing Step by N Debtors')
		# create plot
		sns.barplot(x='Steps',
			        y='Mean Seconds',
			        hue='N Debtors',
			        data=df_for_plot,
			        ax=ax[4])
		# fix overlap
		plt.tight_layout()
		# save
		plt.savefig(str_filename, bbox_inches='tight')
		# save to object
		self.df_output_1 = df_output_1
		self.df_output_2 = df_output_2
		#self.fig = fig
		# return
		return self