import ast
import time
import pandas as pd

# define class
class TimeParsing:
	# initialize
	def __init__(self, ser_payloads):
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
			cls_parse_payload.generate_output(json_str_request=json_str_request)
			# get time in sec
			flt_sec = time.perf_counter() - time_start
			# append to list_flt_sec
			list_flt_sec.append(flt_sec)
			# get int_n_debtors
			int_n_debtors = len(cls_parse_payload.list_unique_id)
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

