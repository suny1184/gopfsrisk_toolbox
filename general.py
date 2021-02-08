# functions
import logging
import pickle

# define function for logging
def LOG_EVENTS(str_filename='./logs/db_pull.log'):
	# set logging format
	FORMAT = '%(name)s:%(levelname)s:%(asctime)s:%(message)s'
	# get logger
	logger = logging.getLogger(__name__)
	# try making log
	try:
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	except FileNotFoundError:
		os.mkdir('./logs')
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	# change to append
	handler = logging.FileHandler(str_filename, mode='a')
	# set the level to info
	handler.setLevel(logging.INFO)
	# set format
	formatter = logging.Formatter(FORMAT)
	# format the handler
	handler.setFormatter(formatter)
	# add handler
	logger.addHandler(handler)
	# return logger
	return logger

# define function for writing to pickle
def PICKLE_TO_FILE(item_to_pickle, str_filename='./output/transformer.pkl', logger=None):
	# pickle file
	pickle.dump(item_to_pickle, open(str_filename, 'wb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')