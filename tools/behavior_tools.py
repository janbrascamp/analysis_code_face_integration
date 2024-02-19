from IPython import embed as shell
from bids import BIDSLayout
import os
import subprocess
import paramiko
import glob
import json
import re
import numpy
import shutil
import copy
import docker
import pandas
import matplotlib
import matplotlib.pyplot as pl
import logging
import inspect

from . import general_tools, BIDS_tools, fMRI_tools

behavior_tools_logger = general_tools.get_logger(__name__)

def assemble_events(raw_events,column_value_pairs,derived_event_names,max_time_s=None):
	"""Create events of a derived type on the basis of the events that are stored in the events.tsv file of an experment. For instance, 
		that tsv file may have a row for each key that was pressed, and in such rows a column that indicates which key it was. Then this
		function allows you to, say, isolate specifically spacebar presses as their own event type, typically for use in a GLM. The function
		also allows you to discard event that fall after max_time_s, which can be useful in cases where some of the events in events.tsv
		may have happened after the scan already ended, and we want to make sure they don't end up in the design matrix.
	
	Arguments:
		raw_events (pandas dataframe). Less-processed events. Column names must include 'trial_type'. 
		column_value_pairs (list of dictionaries). Each dictionary key is a column name featured in raw_events, and its value is the value
			that an entry in that column must have to qualify. The value can also be given as a list if multiple values are allowed.
			A derived event of the kind specified by one dictionary is defined by the conjunction of all values in all columns of
			a given row of raw_events. Each dictionary in column_value_pairs, then, defines one derived event type.
		derived_event_names (list of strings). The names of the derived events. The length of derived_event_names must match that
			of column_value_pairs.
		max_time_s (number,optional). Any events that fall after this moment (in seconds) are discarded.
	
	Returns:
		derived_events (pandas dataframe). Derived events, in the same format as raw_events, so including at least the column names
			onset, duration, and event_type. The event_type column will be filled from derived_event_names, and the other two from raw_events.
	"""
	
	behavior_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	derived_events  = pandas.DataFrame()
	
	for event_index, column_value_pairs_this_event in enumerate(column_value_pairs):
		
		these_events=raw_events.copy()
		
		for one_column_name in column_value_pairs_this_event:
			
			one_column_value_options=column_value_pairs_this_event[one_column_name]
			
			if type(one_column_value_options)==list:	#if multiple values for this column all qualify
			
				these_events=these_events[these_events[one_column_name].isin(one_column_value_options)]
				
			else:
				
				these_events=these_events[these_events[one_column_name]==one_column_value_options]
		
		if max_time_s:
			
			these_events=these_events[these_events['onset']<max_time_s]
				
		these_events['trial_type']=derived_event_names[event_index]
		
		derived_events=derived_events.append(these_events, ignore_index=True)

	return derived_events
