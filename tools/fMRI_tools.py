from IPython import embed as shell
from bids import BIDSLayout
import os
import subprocess
import paramiko
import glob
import json
import re
import numpy as np
import shutil
import copy
import docker
import pandas
import logging
import inspect
import nibabel as nib
import random
import time
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict
from nilearn.datasets import fetch_atlas_harvard_oxford
from ants import from_nibabel, apply_transforms
from ants import image_read as image_read_ants			#give it a diffent name to avoid confusion because 'image_read' is quite generic
from nilearn.glm.first_level import make_first_level_design_matrix, spm_hrf, spm_time_derivative
import matplotlib.pyplot as pl
from  statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from scipy.ndimage import binary_dilation

from . import general_tools, BIDS_tools, behavior_tools

fMRI_tools_logger = general_tools.get_logger(__name__)

def build_models_confounds(models_confounds,nuisance_regressors_included=None):
	"""For a models_confounds object that is returned by the first_level_from_bids() function of nilearn.glm.first_level, create confounds of a derived type.
		A confounds_events object from first_level_from_bids() contains all confounds, and the confound timeseries contain NaNs and are not zero-meaned;
		this function subselects only the confounds defined in nuisance_regressors_included (if defined), and it zero-means and turns NaNs into 0s.
	
	Arguments:
		models_confounds (2D list of pandas dataframes). The two levels of the list of lists are, respectively, subject and run. Each dataframe
			contains all confound timeseries for a subject and run.
		nuisance_regressors_included (list of strings, optional). Nuisance regressors to be present in returned variable. All will be included if not defined.
	
	Returns:
		models_confounds_no_nan (2D list of pandas dataframes). Same nature as models_confounds, but now containing the selected and modified confound time series.

	"""
	
	models_confounds_no_nan = []
	for confounds_one_model in models_confounds:	#for each model
		models_confounds_no_nan_one_model=[]
		for confounds_one_model_and_run in confounds_one_model:	#for each run that is part of that model
			zero_meaned=confounds_one_model_and_run-confounds_one_model_and_run.mean()	#I'm zero-meaning each column here because I'm not sure whether that happens implicitly when model.fit is called
			if nuisance_regressors_included:
				models_confounds_no_nan_one_model.append(zero_meaned.fillna(0)[nuisance_regressors_included])
			else:
				models_confounds_no_nan_one_model.append(zero_meaned.fillna(0))
		models_confounds_no_nan.append(models_confounds_no_nan_one_model)

	return models_confounds_no_nan


def build_models_events(models_events,column_value_pairs,derived_event_names,max_times_s=None):
	"""For a models_events object that is returned by the first_level_from_bids() function of nilearn.glm.first_level, create events of a derived type.
		A models_events object from first_level_from_bids() contains all events stored in the events.tsv files across all subjects and runs included in the model;
		this function creates, across all those levels, new events based on criteria applied to the original events. See behavior_tools.assemble_events() for
		more details: the present function simply applies that function to each subject and run of models_events.
	
	Arguments:
		models_events (2D list of pandas dataframes). The two levels of the list of lists are, respectively, subject and run. Each dataframe
			is exactly the raw_events argument of behavior_tools.assemble_events().
		column_value_pairs (list of dictionaries). Same as argument of the same name in behavior_tools.assemble_events().
		derived_event_names (list of strings). Same as argument of the same name in behavior_tools.assemble_events().
		max_times_s (2D list of numbers, optional). The latest moment within each scan at which events are accepted. In the same order as models_events. 
			If provided, any events that fall after these times will be removed.
		
	
	Returns:
		models_events_of_interest (2D list of pandas dataframes). Same nature as models_events, but now containing only the derived events.

	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])
	
	models_events_of_interest = []
	for model_index,events_one_model in enumerate(models_events):	#for each model
		models_events_of_interest_one_model=[]
		for run_index,events_one_model_and_run in enumerate(events_one_model):	#for each run that is part of that model
		
			if max_times_s:
				this_max_time_s=max_times_s[model_index][run_index]
			else:
				this_max_time_s=None
		
			models_events_of_interest_one_model_and_run=behavior_tools.assemble_events(events_one_model_and_run,column_value_pairs,derived_event_names,this_max_time_s)
			models_events_of_interest_one_model+=[models_events_of_interest_one_model_and_run]
	
		models_events_of_interest+=[models_events_of_interest_one_model]
	
	return models_events_of_interest


def classify_K_fold(voxel_vals_4d,labels,nifti_masker,number_of_splits=5,classifier_name='LinearSVC',classifier_random_state=10, zscore_data = True):
	"""
	Uses k-fold cross-validation to perform classification, based on brains full of voxel values, accompanying labels, as well as a mask to extract voxel values from specific voxels.
		Separately compute classification performance and derive a prediction for each individual sample. Establishing performance based simply on the latter is apparently
		not a good idea (section 3.1.1.2. Obtaining predictions by cross-validation here https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).
	
	Arguments:
	
		voxel_vals_4d (nibabel.nifti1 object): voxel values (for instance beta weights). Eacht time slice corresponds to one entry in 'labels' argument
		labels (numpy array): labels, in the same order as the time slices in voxel_vals_4d
		nifti_masker (nilearn.input_data.NiftiMasker object): mask in the same space as voxel_vals_4d, that specifies which voxels to use.
		number_of_splits (int, optional): number of splits K used for K-fold cross-validation (https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).
		classifier_name (string, optional): which kind of sklearn classifier to use.
		classifier_random_state (integer, optional): the random seed used by classifier object to randomly assign items to folds. Using the same seed on each call ensures
			that results are comparable across calls.
		zscore_data (boolean, optional): whether to z-score the data in voxel_vals_4d (per 3D brain full of betas) before starting the classification procedure
	
	Returns:
		
		score_per_split (Nx1 numpy array): proportion correct per split
		predictions (Nx1 numpy array): classifier predictions, in the same order as labels argument
	
	Example usage:
		
		import nibabel as nib
		from nilearn.input_data import NiftiMasker
		import numpy as np
	
		z_map_thresholded = nib.Nifti1Image(z_data_thresholded, contrast_image.affine, contrast_image.header)
		my_masker = NiftiMasker(mask_img=z_map_thresholded, standardize=False, detrend=False, memory="nilearn_cache", memory_level=2)
		
		my_beta_weights_4d=nib.load(full_path_beta_weight_file)
		my_labels=np.load(full_path_to_label_file)
	
		fMRI_tools.classify_K_fold(my_beta_weights_4d,my_labels,my_masker)
	
	
	Remarks:
	
		Currently using 'balanced_accuracy' as the scoring argument for cross validation. With that measure it is less relevant whether we have equal numbers of samples
		 	in each class. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
	
	
	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	voxel_vals=nifti_masker.fit_transform(voxel_vals_4d)		#turn nifti1 object into numpy array of voxel values that lie within the masked region (dimensions: labels.size, voxels)

	if zscore_data:
		zscored_voxel_pattern = []
		for one_voxel_pattern in voxel_vals:
			average = np.average(one_voxel_pattern)
			sd = np.std(one_voxel_pattern)
			new_voxel = []
			for one_voxel in one_voxel_pattern:
				new_voxel.append((one_voxel - average)/sd)
			zscored_voxel_pattern.append(new_voxel)
		voxel_vals = zscored_voxel_pattern

	if classifier_name=="LinearSVC":
		from sklearn.svm import LinearSVC
		classifier = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=20000, random_state=classifier_random_state)
	else:
		raise Exception("We don't know classifier objects of type "+classifier_name)
	
	score_per_split = cross_val_score(estimator=classifier,	
		X=voxel_vals,
		y=labels,
		cv=number_of_splits,
		scoring='balanced_accuracy',
		n_jobs=-1,
		verbose=1)
	
	predictions = cross_val_predict(estimator=classifier,	
		X=voxel_vals,
		y=labels,
		cv=number_of_splits,
		n_jobs=-1,
		verbose=1)
		
	return [score_per_split,predictions]

		
def classify_leave_1_out(voxel_vals_4d,labels,nifti_masker,classifier_name='LinearSVC',classifier_random_state=10, zscore_data = True):
	"""
	Uses leave-1-out approach to perform classification, based on brains full of voxel values, accompanying labels, as well as a mask to extract voxel values from specific voxels.

	Arguments:
	
		voxel_vals_4d (nibabel.nifti1 object): voxel values (for instance beta weights). Eacht time slice corresponds to one entry in 'labels' argument
		labels (numpy array): labels, in the same order as the time slices in voxel_vals_4d
		nifti_masker (nilearn.input_data.NiftiMasker object): mask in the same space as voxel_vals_4d, that specifies which voxels to use.
		classifier_name (string, optional): which kind of sklearn classifier to use.
		classifier_random_state (integer, optional): the random seed used by classifier object to randomly assign items to folds. Using the same seed on each call ensures
			that results are comparable across calls.
		zscore_data (boolean, optional): whether to z-score the data in voxel_vals_4d (per 3D brain full of betas) before starting the classification procedure
	
	Returns:
		
		score_per_split (Nx1 numpy array): proportion correct (either 1 or 0) per item
	
	Example usage:
		
		See fMRI_tools.classify_K_fold() for example usage of that function, which is very similar to this one.
	
	"""
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	voxel_vals=nifti_masker.fit_transform(voxel_vals_4d)		#turn nifti1 object into numpy array of voxel values that lie within the masked region (dimensions: labels.size, voxels)

	if zscore_data:
		zscored_voxel_pattern = []
		for one_voxel_pattern in voxel_vals:
			average = np.average(one_voxel_pattern)
			sd = np.std(one_voxel_pattern)
			new_voxel = []
			for one_voxel in one_voxel_pattern:
				new_voxel.append((one_voxel - average)/sd)
			zscored_voxel_pattern.append(new_voxel)
		voxel_vals = zscored_voxel_pattern

	groups=np.array(range(len(labels)))				#each voxel pattern its own chunk
	
	if classifier_name=="LinearSVC":
		from sklearn.svm import LinearSVC
		classifier = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=20000, random_state=classifier_random_state)
	else:
		raise Exception("We don't know classifier objects of type "+classifier_name)
	
	score_per_split = cross_val_score(estimator=classifier,
		X=voxel_vals,
		y=labels,
		groups=groups,
		cv=LeaveOneGroupOut(),
		n_jobs=-1,
		verbose=1)

	predictions = cross_val_predict(estimator=classifier,	
		X=voxel_vals,
		y=labels,
		groups = groups,
		cv=LeaveOneGroupOut(),
		n_jobs=-1,
		verbose=1)

	return [score_per_split,predictions]
	

def classify_test_from_train(train_weights_4d,train_labels,test_weights_4d,test_labels,mask, classifier_name='LinearSVC',classifier_random_state=10, zscore_data = True):
	"""
	Perform classification on a set of data using a classifier that has been trained on a different, nonoverlapping set of data. For example, decoding main data from a classifier trained on localizer data. The inputs are brains full of voxel values, accompanying labels, as well as a mask to extract voxel values from specific voxels.

	Arguments:
	
		train_weights_4d, test_weights_4d (nibabel.nifti1 object): voxel values (for instance beta weights) Each time slice corresponds to one entry in 'labels' argument
		train_labels, test_labels (numpy array): trainig labels, in the same order as the time slices in voxel_vals_4d
		mask (nilearn.input_data.NiftiMasker object): mask in the same space as voxel_vals_4d, that specifies which voxels to use.
		classifier_name (string, optional): which kind of sklearn classifier to use.
		classifier_random_state (integer, optional): the random seed used by classifier object to randomly assign items to folds. Using the same seed on each call ensures
			that results are comparable across calls.
		zscore_data (boolean, optional): whether to z-score the data in train_weights_4d and test_weights_4d (per 3D brain full of betas) before starting the classification procedure
	
	Returns:
		
		accuracy: proportion correct
		prediction (Nx1 numpy array): classifier predictions, in the same order as labels argument
	
	Example usage:
		
		Decoding main data from a classifier trained on localizer data.
	
	"""
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	train_voxel_vals=mask.fit_transform(train_weights_4d)
	test_voxel_vals=mask.fit_transform(test_weights_4d)

	if zscore_data:
		""" Zscore the training data. """
		new_train_voxel_vals = []
		for one_train_voxel_pattern in train_voxel_vals:
			train_average = np.average(one_train_voxel_pattern)
			train_sd = np.std(one_train_voxel_pattern)
			new_train_voxel_pattern = []
			for one_train_voxel in one_train_voxel_pattern:
				new_train_voxel_pattern.append((one_train_voxel - train_average)/train_sd)
			new_train_voxel_vals.append(new_train_voxel_pattern)
		train_voxel_vals = new_train_voxel_vals

		""" Zscore the to-be-classified data. """
		new_test_voxel_vals = []
		for one_test_voxel_pattern in test_voxel_vals:
			test_average = np.average(one_test_voxel_pattern)
			test_sd = np.std(one_test_voxel_pattern)
			new_test_voxel_pattern = []
			for one_test_voxel in one_test_voxel_pattern:
				new_test_voxel_pattern.append((one_test_voxel - test_average)/test_sd)
			new_test_voxel_vals.append(new_test_voxel_pattern)
		test_voxel_vals = new_test_voxel_vals
	
	if classifier_name=="LinearSVC":
		from sklearn.svm import LinearSVC
		classifier = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=20000, random_state=classifier_random_state)
	else:
		raise Exception("We don't know classifier objects of type "+classifier_name)
				
	# train model
	classifier.fit(train_voxel_vals, train_labels)
	prediction = classifier.predict(test_voxel_vals)
	accuracy = classifier.score(test_voxel_vals,test_labels)
	return [accuracy, prediction]


def concatenate_design_matrices(matrices):
	"""
	Concatenate design matrices row-wise, with time counting on cumulatively from one design matrix to the next. Any columns with a column name that is present for one but not all
		contributing matrices are, in the resulting matrix, assigned a 0 on all rows that correspond to a contributing matrix that lacks that column name. Any columns that share the
		same name across multiple contributing matrices will be represented in a single column in the resulting matrix.
	
	Arguments:
	
		matrices (list of pandas dataframes): each dataframe corresponds to one design matrix. The rows (.index) of each dataframe should indicate time; the column names (.columns)
			should indicate regressor names.

	Returns:
		
		matrix_concatenated (pandas dataframe): the concatenated matrix.
	
	"""
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])
	
	matrices=copy.deepcopy(matrices)		#make deepcopy not to edit the argument variable in the calling function when we change the .index values
	
	"""
	Make row values (.index) count cumulatively from one element of matrices to the next
	"""
	for matrix_index in range(1,len(matrices)):
		matrices[matrix_index].index+=matrices[matrix_index-1].index[-1]
	
	matrix_concatenated=pandas.concat(matrices)	#Concatenate matrices. This automatically puts columns that share the same name across elements of 'matrices' in the same column of matrix_concatenated,
												#and puts NaN values on the appropriate rows if a given column is not defined in a particular element of 'matrices'.
	
	matrix_concatenated=matrix_concatenated.fillna(0)	#Put 0s at all those NaN places where a given column was not defined.
	
	return matrix_concatenated


def make_first_level_design_matrix_one_column_at_a_time(frame_times, events, hrf_model, drift_model=None, high_pass=None, drift_order=None,add_regs=None):
	"""
	Make a first level design matrix using nilearn.glm.first_level.make_first_level_design_matrix(), but call that method one event at a time and horizontally stack the resulting columns together. This should avoid the warning
		'Matrix is singular at working precision, regularizing...' and the associated action, on the part of nilearn, to regularize the matrix. This is a workaround because nilearn.glm.first_level.make_first_level_design_matrix()
		does not allow control over whether the regularization happens.
	
	Arguments (same as nilearn.glm.first_level.make_first_level_design_matrix(); see there for interpretation: https://nilearn.github.io/modules/generated/nilearn.glm.first_level.make_first_level_design_matrix.html):
	
		frame_times (numpy array): times of individual time slices in seconds; see make_first_level_design_matrix documentation.
		events (pandas DataFrame): the events that are to featured in the design matrix; see make_first_level_design_matrix documentation.
		hrf_model (string or other formats): the hemodynamic response function model; see make_first_level_design_matrix documentation.
		drift_model (string, optional):	model of slow drift; see make_first_level_design_matrix documentation.
		high_pass (float, optional): high-pass frequency that determines what frequency range is covered by the drift model; see make_first_level_design_matrix documentation.
		drift_order (int, optional): polynomial order that determines flexibility of the drift model; see make_first_level_design_matrix documentation.
		add_regs (pandas DataFrame): (nuisance) regressors, which should not be convolved with the HRF; see make_first_level_design_matrix documentation.
		
	Returns:
		complete_design_matrix (pandas DataFrame): design matrix including all events, slow drift regressors and nuisance regressors
	
	Notes: it is not a good idea to use this method instead of nilearn.glm.first_level.make_first_level_design_matrix() if the resulting design matrix will be entered into a GLM: nilearn performs the regularization for a reason.
		However, if the design matrix is used for other purposes, for instance if it will be concatenated with other design matrices into a larger matrix for use in a GLM, then avoiding regularization right here can make sense.
		In such cases it is a good idea to check the condition number of the matrix that you do want to use in a GLM, in the same way nilearn.glm.first_level.make_first_level_design_matrix() does it, on line 388 here:
		https://github.com/nilearn/nilearn/blob/9f5ae944449dbfded6bfcd34e704ad8662ead45f/nilearn/glm/first_level/design_matrix.py

	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])
	
	trial_type_options=list(set(events['trial_type']))
	trial_type_options.sort()
	
	complete_design_matrix=None
	
	""" 1. add, one by one, the columns for the individual task events in 'events' """
	
	for trial_type_option in trial_type_options:
		events_of_only_this_kind=events.loc[events['trial_type']==trial_type_option]
		design_matrix_single_event=make_first_level_design_matrix(frame_times,events=events_of_only_this_kind, hrf_model=hrf_model, drift_model=None)	#avoid drift regressors explicitly by setting drift_model to None, and avoid confound regressors by not passing add_regs
		
		if 'constant' in design_matrix_single_event:
			del design_matrix_single_event['constant']		#by default nilearn.glm.first_level.make_first_level_design_matrix adds a 'constant' column filled with ones for each design_matrix_single_event. We don't need all of those: we get one when we add the drift model.
			
		if complete_design_matrix is not None:
			
			complete_design_matrix=pandas.concat([complete_design_matrix, design_matrix_single_event], axis=1)
			
		else:
			
			complete_design_matrix=design_matrix_single_event
			
	""" 2. add, all at once, the columns for the nuisance regressors in 'add_regs' """
	
	if add_regs is not None:
		
		if complete_design_matrix is not None:
			
			complete_design_matrix=pandas.concat([complete_design_matrix, add_regs.set_index(frame_times)], axis=1)
		
		else:
			
			complete_design_matrix=add_regs.set_index(frame_times)
	
	""" 3. add, all at once, the columns for the slow drift defined by 'drift model', 'high_pass' and 'drift_oder' """
	
	if drift_model:
		
		design_matrix_drift=make_first_level_design_matrix(frame_times, hrf_model=hrf_model, drift_model=drift_model, high_pass=high_pass, drift_order=drift_order)
		
		if complete_design_matrix is not None:
			
			complete_design_matrix=pandas.concat([complete_design_matrix, design_matrix_drift], axis=1)
		
		else:
			
			complete_design_matrix=design_matrix_drift
			
	return complete_design_matrix
	

def open_in_MRIcroGL(base_image_path,overlay_info=[],short_identifier='',run_option='now',mountpoint_local_remote=None):
	"""Provide functionality to open an image file in MRIcroGL, and to optionally add overlays. Depending on the value of 'run_option', either the image will be opened
		from within Python via subprocess.Popen, or command prompt code for opening the image will be written to a text file. The latter option is helpful, for instance,
		if you're running this Python code on a server yet would like to open the image locally.
	
	Arguments:
		base_image_path (string): absolute path of image file.
		overlay_info (Nx3 list; optional): if given, each element in this list defines one overlay and has 3 components: the path of the overlay file (string), the color map (string that's valid for a color map 
			in MRIcroGL), and the lower and upper overlay values to be shown (1x2 list of floats). The overlay is not shown outside of this range.
		short_identifier (string; optional): a string that concisely describes what these data are about. If run_option='now', then this name will appear in the MRIcroGL interface as the name of the first
			volume shown. If run_option="later" then this name will be used for the saved python script.
		run_option (string; optional): if 'now',  then MRIcroGL will be called from within this function to open the image right now. If 'later', then a python script will be generated instead, with code that can
			run by MRIcroGL later and/or from a different computer, to open the image at that time, by typing MRIcroGL [path to script file] at a command prompt. The script will be put in a 'scripts' folder inside the
			same 'derivatives' subdirectory that overlay_info[0] is in, except if there are no overlays, in which case the 'derivatives' subdirectory of base_image_path is used.
		mountpoint_local_remote (2x1 list of strings; optional): if run_option is 'later', and if you plan to run the resulting script from a local machine on which the file system containing the MRI data have been mounted, then
			you can enter values here to specify 1) the absolute path of the mountpoint on the local machine and 2) the mounted directory on the remote machine. It is assumed that base_image_path and any paths in
			overlay_info are all within the path indicated by mountpoint_local_remote[1].
	
	Returns:
		Nothing. But opens MRIcroGL with the requested image and optional overlay(s).
	
	Remarks:
		-The run_option='now' variant is currently not using a python script even though it might, in which case the code for it can be integrated much more with the run_option='later' variant
	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	if run_option=='now':
	
		shell_command_arguments=['MRIcroGL']
	
		if short_identifier:		#hacky approach to making short_identifier appear in MRIcroGL interface: copy the image at base_image_path to a file with the filename specified by short_identifier, and show that one as the first image, with data range set so that it doesn't show anything.
		
			[base_image_location,base_image_filename]=os.path.split(base_image_path)
			base_image_extension=base_image_filename[[index for index,value in enumerate(base_image_filename) if value=='.'][0]:]
		
			path_to_temporary_shortname_file=os.path.join(base_image_location,short_identifier)+base_image_extension
			subprocess.Popen(['cp',base_image_path,path_to_temporary_shortname_file])
			shell_command_arguments+=[path_to_temporary_shortname_file]#,'-dr','100000','100000']
	
		shell_command_arguments+=[base_image_path]
	
		for [overlay_path,color_map_string,min_max_vals] in overlay_info:
		
			shell_command_arguments+=[overlay_path,'-cm',color_map_string,'-dr']
			shell_command_arguments+=[str(value) for value in min_max_vals]
	
		process_of_opening_MRIcroGL = subprocess.Popen(shell_command_arguments)
		#subprocess.Popen([sys.executable, '-c']+shell_command_arguments)
		#subprocess.Popen(theunit,shell=True)
	
		if short_identifier:
			time.sleep(2.)	#give MRIcroGL some time to process the command before removing the file it has to use. Tried better ways that included monitoring the output of the process to see its progress, but that turned out difficult.
			subprocess.Popen(['rm',path_to_temporary_shortname_file])

	elif run_option=='later':
		#the following code draws on https://www.nitrc.org/plugins/mwiki/index.php/mricrogl:MainPage#Scripting and https://github.com/neurolabusc/MRIcroGL10_OLD/blob/master/COMMANDS.md
		
		script_lines=["# run this file by typing 'MRIcroGL [path_to_this_file] or by dragging the file onto the MRIcroGL dock icon","# the latter option doesn't allow multiple instances of MRIcroGL to be open at the same time",'import gl','gl.resetdefaults()']
		
		if mountpoint_local_remote:
			script_lines+=["gl.loadimage('"+mountpoint_local_remote[0].rstrip('/')+base_image_path[len(mountpoint_local_remote[1].rstrip('/')):]+"')"]
		else:
			script_lines+=["gl.loadimage('"+base_image_path+"')"]
		
		for overlay_index, [overlay_path,color_map_string,min_max_vals] in enumerate(overlay_info):
			if mountpoint_local_remote:
				script_lines+=["gl.overlayload('"+mountpoint_local_remote[0].rstrip('/')+overlay_path[len(mountpoint_local_remote[1].rstrip('/')):]+"')"]
			else:
				script_lines+=["gl.overlayload('"+overlay_path+"')"]

			script_lines+=['gl.minmax('+str(overlay_index+1)+', '+str(min_max_vals[0])+', '+str(min_max_vals[1])+')']
			script_lines+=['gl.colorname('+str(overlay_index+1)+",'"+color_map_string+"')"]
			
		script_lines+=['gl.orthoviewmm(0,0,0)']
		
		base_image_location=os.path.split(base_image_path)[0]
		
		if overlay_info:
			path_of_imaging_data=overlay_info[0][0]
		else:
			path_of_imaging_data=base_image_location
			
		script_directory_components=[]
		for component in path_of_imaging_data.split('/'):
			if component in ['anat','func','temp']:
				script_directory_components+=['scripts']
				break
			script_directory_components+=[component]
			
		script_directory='/'.join(script_directory_components)
		os.makedirs(script_directory, exist_ok=True)
		
		script_path=os.path.join(script_directory,'run_me_'+short_identifier+'.py')
		
		with open(script_path, 'w') as f:
			for line in script_lines:
				f.write(line+'\n')

def plot_vif_and_correlation_matrix(design_matrix, vif_filepath, corrmatt_filepath, plot_title=''):
	"""Produce two plots that inform about the nature of a design matrix: the variance inflation factor (VIF) and the correlation matrix among regressors.
	
	Arguments:
		design_matrix (design matrix object from nilearn.glm.first_level): a design matrix.
		vif_filepath (string): absolute path of plot that will show VIF info.
		corrmatt_filepath (string): absolute path of plot that will show correlation matrix.
		plot_title (string, optional): what to print at the top of both plots.
	
	Returns:
		Nothing, but does save 2 plots to file.
	"""
	
	pl.figure(figsize=(30,25))
	my_cols = design_matrix.columns
	my_vars = []
	for column in range(len(my_cols)):
		my_vars.append(variance_inflation_factor(design_matrix,column))

	
	# pl.yscale('log')
	pl.axes(yscale = 'log', ylim = (0,10000))
	pl.plot(my_cols,my_vars)
	pl.xticks(rotation = 90, fontsize = 10)
	pl.title(plot_title, fontsize = 30)
	pl.savefig(vif_filepath)

	pl.figure(figsize=(30,25))
	sns.heatmap(design_matrix.corr(),vmin = -1, vmax = 1,cmap = "vlag")
	pl.xticks(rotation = 90, fontsize = 10)
	pl.yticks(fontsize = 10)
	pl.title(plot_title, fontsize = 30)
	pl.savefig(corrmatt_filepath)

def put_freesurfer_label_in_T1_space(fMRIPrep_derivatives_folder,output_derivatives_folder,subject,labels):
	"""Take a freesurfer label defined in a subject's Freesurfer 'label' folder, put it in the space of the subject's T1 anatomical that Freesurfer used,
		and load and return it as a nibabel nifti image object. This function assumes that the Freesurfer segmentation etc. were produced as part of fMRIPrep, and
		can therefore be found in the fMRIPrep derivatives folder. If multiple labels are provided, a list of multiple nifti image objects is returned.
	
	Arguments:
		fMRIPrep_derivatives_folder (string). Absolute path of derivatives folder that contains fMRIPrep result. In this folder 'sourcedata/freesurfer/
			can be found.
		output_derivatives_folder (string). Absolute path of output derivatives folder in which the file in anatomical space will be placed.
		subject (string): identifier of the subject, not including the 'sub-' part.
		labels (string or list of strings): name(s) of the label file(s), not including the extension '.label'. Can be either a single string or a list of multiple.
	
	Returns:
		nibabel nifti image object containing the label data (such that each voxel within the label has a 1 and the rest a 0),
			or list of nifti image objects in case multiple labels were passed.
		Also saves per-label nifti's to the subject's 'anat' path in output_derivatives_folder.
	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])
	
	if type(labels)==str:
		labels=[labels]
		return_as_list=False
	else:
		return_as_list=True	
		
	nifti_imgs=[]
	for label in labels:
		
		input_full_path=os.path.join(fMRIPrep_derivatives_folder,'sourcedata','freesurfer','sub-'+subject,'label',label+'.label')
	
		output_anat_path=os.path.join(output_derivatives_folder,'sub-'+subject,'anat')
		os.makedirs(output_anat_path, exist_ok=True)
	
		label_camel_case=''.join([element.title() for element in re.split(r'_|\.', label)]) #convert label name to camel case because underscores mean something in BIDS filenames, and so do periods
		
		entities = {
			'sub': subject,
			'space': 'T1w',
			'desc': label_camel_case,
			'suffix': 'mask'
		}
		output_filename=BIDS_tools.my_build_bids_filename(entities)
		output_full_path=os.path.join(output_anat_path,output_filename)
	
		template_full_path=os.path.join(fMRIPrep_derivatives_folder,'sourcedata','freesurfer','sub-'+subject,'mri','orig.mgz')
		
		#first convert the label (in surface space) to a nii.gz file (in T1 space) and store it in the right place.
		shell_command_arguments=['mri_label2vol','--label',input_full_path,'--temp',template_full_path,'--identity','--o',output_full_path]
		general_tools.execute_shell_command(shell_command_arguments,collect_response=False,wait_to_finish=True)
	
		#then load that file as a nibabel nifti image object.
		nifti_imgs+=[nib.load(output_full_path)]
	
	if return_as_list:
		return nifti_imgs
	else:
		return nifti_imgs[0]



def put_harvard_oxford_label_in_T1_space(fMRIPrep_derivatives_folder,output_derivatives_folder,subject,labels,atlas_folder=None,atlas_name=None,layout=None):
	"""Take a label defined in a 'max probability' (i.e. not probabilistic) map of the Harvard-Oxford FSL parcellation, put it in the space of the subject's T1 anatomical that is
		in the subject's anat folder in fMRIPrep_derivatives, and return it as a nibabel nifti image object. If multiple labels are provided, a list of multiple nifti image objects is returned.
	
	Arguments:
		fMRIPrep_derivatives_folder (string). Absolute path of derivatives folder that contains fMRIPrep result.
		output_derivatives_folder (string). Absolute path of output derivatives folder in which the file in anatomical space will be placed.
		subject (string): identifier of the subject, not including the 'sub-' part.
		labels (string or list of strings): name(s) of the labels as provided by the 'labels' field of the object returned by fetch_atlas_harvard_oxford(). Can be either a single string or a list of multiple.
		atlas_folder (string, optional). Absolute path of folder into which the entire Harvard Oxford atlas will be fetched, or will be found if it has already been fetched on a previous occasion.
			If not provided, then location of 'sourcedata' folder will be guessed on the basis of fMRIPrep_derivatives_folder, and used.
		atlas_name (string, optional). The name of one of the 'maxprob' atlases, as fetched by nilearn.datasets.fetch_atlas_harvard_oxford.
		layout (BIDSLayout object, optional). Created with 'rawdata' folder as first argument and fMRIPrep derivatives folder as named 'derivatives' argument. If not provided, then location of 'rawdata'
			will be guessed based on fMRIPrep_derivatives_folder, and layout will be created within the function. If you have this variable in the calling function, then it saves time to pass it.
	
	Returns:
		nibabel nifti image object containing the label data (such that each voxel within the label has a 1 and the rest a 0),
			or list of nifti image objects in case multiple labels were passed.
		Also saves per-label nifti's to the subject's 'anat' path in output_derivatives_folder.
	
	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])
	
	if type(labels)==str:
		labels=[labels]
		return_as_list=False
	else:
		return_as_list=True
	
	if not(atlas_folder):
		atlas_folder=os.path.join(re.search('(.*)derivatives.*',fMRIPrep_derivatives_folder)[1],'sourcedata')
		
	if not(atlas_name):
		atlas_name='cort-maxprob-thr25-2mm'
		
	if not(layout):
		rawdata_folder=os.path.join(re.search('(.*)derivatives.*',fMRIPrep_derivatives_folder)[1],'rawdata')
		layout = BIDSLayout(rawdata_folder,derivatives=fMRIPrep_derivatives_folder)
	
	label_info = fetch_atlas_harvard_oxford(atlas_name, data_dir=atlas_folder)		#see https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html
	label_names_in_order=label_info['labels']

	labeled_brain_MNI_space_ants = from_nibabel(label_info['maps'])		#label_info['maps'] is a nibabel nii object, but for the transformation to T1 we use a method that requires an ants object instead, so we convert.
	
	anat_path=layout.get(subject=subject,desc='preproc',suffix='T1w',space=None,extension='nii.gz',scope='derivatives',return_type='filename')[0]
	t1_reference_image_ants=image_read_ants(anat_path)

	transforms=[[candidate for candidate in layout.get(subject=subject,to='T1w',extension='h5',scope='derivatives',return_type='filename') if 'from-MNI' in candidate][0]]	#determine that 'from-MNI' part outside of the layout.get method because using 'from'
																																											#as a named argument throws a syntax error
	
	labeled_brain_T1w_space_ants = apply_transforms(t1_reference_image_ants,labeled_brain_MNI_space_ants,transforms,interpolator = 'nearestNeighbor' ) 	#see https://brainhack-princeton.github.io/handbook/content_pages/04-03-registration.html, https://pypi.org/project/antspyx/, and
																																						#https://github.com/stnava/structuralFunctionalJointRegistration/blob/master/src/Default%20Mode%20Connectivity%20in%20ANTsPy.ipynb

	labeled_brain_T1w_space_nibabel=labeled_brain_T1w_space_ants.to_nibabel() #labeled_brain_T1w_space_ants is an ants object, but downstream we want a nibabel object, so we convert.
	
	output_anat_path=os.path.join(output_derivatives_folder,'sub-'+subject,'anat')
	os.makedirs(output_anat_path, exist_ok=True)
	
	entities = {
		'sub': subject,
		'space': 'T1w',
		'suffix': 'mask'
	}
	
	return_brains=[]
	for one_label in labels:
		
		this_index=[index for index,label_name in enumerate(label_names_in_order) if label_name==one_label][0]
		boolean_data = np.array(labeled_brain_T1w_space_nibabel.get_fdata()==this_index, dtype=np.int8)			#any voxel with the integer value that designates this label gets a 1; all others a 0
		mask_nifti=nib.Nifti1Image(boolean_data, labeled_brain_T1w_space_nibabel.affine, labeled_brain_T1w_space_nibabel.header)
		
		return_brains += [mask_nifti]	#we worked with a numpy array, which we now stick into a nibabel NIfTI object again, with the familiar affine and header.
		
		#also store the mask in T1w space in the derivatives folder, in addition to saving it in return_brains:
		label_camel_case=''.join([element.title() for element in re.split(r' |, |\(|\)', one_label)])
		entities['desc']=label_camel_case
		output_filename=BIDS_tools.my_build_bids_filename(entities)
		output_full_path=os.path.join(output_anat_path,output_filename)
		mask_nifti.to_filename(output_full_path)
		
	if return_as_list:
		return return_brains
	else:
		return return_brains[0]
		
		

def run_fMRIPrep(base_of_BIDS_tree,subjects=[],output_subfolder='',wait_to_finish=True,path_to_freesurfer_license='/Applications/freesurfer/license.txt',output_spaces=[],use_podman=False,use_nohup=False,run_option='now', skip_if_present=True):
	
	"""Run fMRIPrep via the command line call, using either the fmriprep-docker wrapper or podman, on selected subjects in rawdata folder.
	
	Arguments:
		base_of_BIDS_tree (string). Path of the base of the BIDS structure, directly in which 'rawdata' folder is present, which will be passed as input to fMRIPrep.
		subjects (list of strings, optional). Subject identifiers, as used in the 'rawdata' folder (needs to be without the 'sub-' part itself),
			defining which subjects will be analyzed. If not given, then all subjects will be analyzed.
		output_subfolder (string, optional). Name of subfolder inside 'derivatives' where results should be written. If not given, then
			results will be written straight to 'derivatives' without a subfolder.
		wait_to_finish (bool, optional). If True then code will halt till fmriprep-docker is done. If False it won't. Ignored if run_option='later'.
		path_to_freesurfer_license (string, optional). Absolute path of Freesurfer license file that fmriprep needs in order to run.
		output_spaces (list of strings, optional). Output spaces to pass to the '--output_spaces' argument of fmriprep. Determines the spaces of the
			functional output data. Examples: 'MNI152NLin2009cAsym', 'T1w' 'fsaverage5'.
		use_podman (bool, optional): Whether to use podman and use a syntax very similar to calling docker directly in the command line. If false, then the fmriprep-docker
			wrapper will be used. (https://www.nipreps.org/apps/docker/)
		use_nohup (bool,optional): Whether to use nohup. If True, then this function produces a terminal command that is disconnected from the terminal window: the terminal command that would have been
			used with use_nohup=False, is instead embedded within a larger call to nohup. When running on server, this disconnecting from the terminal window makes it so that the process continues regardless
		 	of whether the ssh session stays alive.
		run_option (string, optional): if 'now',  then fMRIPrep will be called from within this function in Python. If 'later', then a single line of command-line script will be
			written to a text file, so that it can run to call fMRIPrep later. The reason is that there have been some issues trying to call fMRIPrep from within python (see 'To do'). The script will be put in a 'scripts' folder inside output_subfolder.
		skip_if_present (boolean, optional): applies only if participant_or_group is 'participant'. If set to True, then the 'subject' list (either passed as an argument or
			created within this function) gets reduced to only those subjects who don't have MRIQC data in the output folder already.
			
	Returns:
		Nothing.
	
	To do:
		It will probably be nice to include ways of controlling other arguments of fmriprep as well.
		I (Jan) ran into issues calling fMRIPrep from within Python, perhaps because it expects an interactive environment or something. Haven't looked into it closely, but
			would be easier if we don't have to do the run_option='later' thing. Vasili can run his command line calls from Python using: 
			proc = subprocess.Popen([shell_command], stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
			(out,err) = proc.communicate()
			which is a different set of choices than are being made by general_tools.execute_shell_command() so in the future it may be worth copying some of his choices
			to get run_option='now' to work.
		The code currently skips participants for whom the number of already fMRIPrep'ed sessions is equal to the number of sessions in rawdata. But I don't know how fMRIPrep deals with
			a situation where new sessions get added to rawdata, after fMRIPrep has already been called on the earlier sessions. I'm guessing it should deal with that OK, because the sessions
			are numbered. 
	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	max_text_file_name_length=50
	
	if output_subfolder:
		output_folder=os.path.join(base_of_BIDS_tree,'derivatives',output_subfolder)
	else:
		output_folder=os.path.join(base_of_BIDS_tree,'derivatives')
		
	os.makedirs(output_folder, exist_ok=True)
	
	layout = BIDSLayout(os.path.join(base_of_BIDS_tree,'rawdata'))
	all_subjects_in_layout=layout.get_subjects()
	
	subjects_to_be_skipped_because_already_present=[]
	
	if skip_if_present:		#skip those subjects for whom all available sessions have already been MRIQC'ed. This assumes that sessions that used to be available in rawdata, remain available.
		
		num_sessions_to_be_analyzed_per_subject=[len(layout.get_sessions(subject=one_sub)) for one_sub in all_subjects_in_layout]
		
		all_subject_folders_in_output=[os.path.split(candidate)[1] for candidate in glob.glob(os.path.join(output_folder,'sub-*')) if os.path.isdir(candidate)]
		for sub_index,one_sub in enumerate(all_subjects_in_layout):
			if 'sub-'+one_sub in all_subject_folders_in_output:
				num_sessions_analyzed_this_subject=len(glob.glob(os.path.join(output_folder,'sub-'+one_sub,'ses-*')))
			else:
				num_sessions_analyzed_this_subject=0
		
			if num_sessions_analyzed_this_subject==num_sessions_to_be_analyzed_per_subject[sub_index]:
				fMRI_tools_logger.info('Skipping '+'sub-'+one_sub+' because fMRIPrep output is already present for a number of sessions that is equal to all sessions available in rawdata for this participant.')
				subjects_to_be_skipped_because_already_present+=[one_sub]
	
	if use_nohup:
		shell_command_arguments=['nohup']
	else:
		shell_command_arguments=[]
		
	if 	use_podman:
		shell_command_arguments+=['podman','run','-it','--rm',\
			'-v',os.path.join(base_of_BIDS_tree,'rawdata')+':/data:ro',\
			'-v',output_folder+':/out',\
			'-v',path_to_freesurfer_license+':/opt/freesurfer/license.txt:ro',\
			'nipreps/fmriprep:latest','/data','/out']	#these are the call to podman and its arguments
	else:
		shell_command_arguments+=['fmriprep-docker',os.path.join(base_of_BIDS_tree,'rawdata'),output_folder]	#these are the call to fmriprep-docker and its arguments
		
	#regardless of use_podman, beyond this are all fmriprep arguments (not fmriprep-docker arguments or docker arguments) so that part is almost identical for both use_podman options
		
	shell_command_arguments+=['participant','--participant-label']
		
	if not(subjects):
		subjects=all_subjects_in_layout
	
	subjects_included=[one_sub for one_sub in subjects if not one_sub in subjects_to_be_skipped_because_already_present]
	shell_command_arguments+=subjects_included
	
	if use_podman:
		shell_command_arguments+=['--fs-license-file','/opt/freesurfer/license.txt']
	else:
		shell_command_arguments+=['--fs-license-file',path_to_freesurfer_license]
	
	shell_command_arguments+=['--output-layout','bids']
	
	if output_spaces:
		shell_command_arguments+=['--output-spaces']
		shell_command_arguments+=output_spaces
	
	if use_nohup:
		os.makedirs(os.path.join(output_folder,'nohup_logs'), exist_ok=True)
		timestamp = time.strftime("%d%b%Y-%H%M%S",time.localtime())
		nohup_output_path=os.path.join(output_folder,'nohup_logs',inspect.stack()[0][3]+'_'+timestamp+'.out')
		shell_command_arguments+=['&>',nohup_output_path,'&']		#nohup_output_path points to a file that keeps track of progress, warnings, etc. Also, calling 'lsof [nohup_output_path]' in the terminal tells you who all is interacting with that file, which allows you to see when the process is done.
	
	if len(subjects_included)==0:
		fMRI_tools_logger.info('On second thoughts: not running fMRIPrep at all because output is already present for all subjects and sessions that are available in rawdata.')
	else:	
		if run_option=='now':
			general_tools.execute_shell_command(shell_command_arguments,collect_response=False,wait_to_finish=wait_to_finish)
		else:
			os.makedirs(os.path.join(output_folder,'scripts'), exist_ok=True)
		
			script_text_file_name_no_suffix='run_me_'+'_'.join(subjects)
		
			if len(script_text_file_name_no_suffix)>max_text_file_name_length:
				suffix='_abbrev'
				script_text_file_name_no_suffix=script_text_file_name_no_suffix[:max_text_file_name_length-len(suffix)]+suffix
			script_text_file_name=script_text_file_name_no_suffix+'.txt'

			script_path=os.path.join(output_folder,'scripts',script_text_file_name)
			with open(script_path, 'w') as f:
				f.write('Copy the following line and paste it in a terminal to run fMRIPrep: ')
				f.write(' '.join(shell_command_arguments))


def run_MRIQC(base_of_BIDS_tree,participant_or_group='participant',subjects=[],output_subfolder='',wait_to_finish=True,use_podman=False,run_option='now', skip_if_present=True):
	
	"""Run MRIQC via Docker, on selected subjects in rawdata folder. This will run in the background so don't expect the results to be done right away,
		in case other processes need them or something.
	
	Arguments:
		base_of_BIDS_tree (string). Path of the base of the BIDS structure, directly in which 'rawdata' folder is present, which will be passed as input to MRIQC.
		participant_or_group (string, either 'participant' or 'group', optional). If 'participant' then will run participant level analysis, if 'group' then group
	`		level analysis.
		subjects (list of strings, optional). Subject identifiers, as used in the 'rawdata' folder (either with or without the 'sub-' part itself),
			defining which subjects (called 'participants' by MRIQC) will be considered if this is a participant-level analysis. If not given, then all subjects will be considered.
			If this is a group-level analysis, then this argument is ignored.
		output_subfolder (string, optional). Name of subfolder inside 'derivatives' where results should be written. If not given, then
			results will be written straight to 'derivatives' without a subfolder.
		wait_to_finish (bool, optional). If True then code will halt till fmriprep-docker is done. If False it won't. Ignored if run_option='later'.
		use_podman (bool, optional): Whether to use podman and use a syntax very similar to calling docker directly in the command line. If false, then 
			python's 'docker' package will be used.
		run_option (string, optional): if 'now',  then MRIQC will be called from within this function in Python. If 'later', then a single line of command-line script will be
			written to a text file, so that it can run to call MRIQC later. The reason is that there have been some issues trying to call fMRIPrep from within python (see 'To do'), 
			and I am unsure whether the same applies to MRIQC. The script will be put in a 'scripts' folder inside output_subfolder. Ignored if use_podman=False.
		skip_if_present (boolean, optional): applies only if participant_or_group is 'participant'. If set to True, then the 'subject' list (either passed as an argument or
			created within this function) gets reduced to only those subjects who don't have MRIQC data in the output folder already.
	
	Returns:
		Nothing.
	
	To do:
		It will probably be nice to include ways of controlling other arguments of mriqc as well, such as which sessions to include.
		Verify whether run_option='now' works here (I don't think it does for run_fMRIPrep, but it may for run_MRIQC). If it doesn't, then consider resolving that.
		The code currently skips participants for whom the number of already MRIQC'ed sessions is equal to the number of sessions in rawdata. But I don't know how MRIQC deals with
			a situation where new sessions get added to rawdata, after MRIQC has already been called on the earlier sessions. I'm guessing it should deal with that OK, because the sessions
			are numbered. 
	
	"""
	
	fMRI_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	max_text_file_name_length=50
	
	if output_subfolder:
		output_folder=os.path.join(base_of_BIDS_tree,'derivatives',output_subfolder)
	else:
		output_folder=os.path.join(base_of_BIDS_tree,'derivatives')
		
	os.makedirs(output_folder, exist_ok=True)
	
	layout = BIDSLayout(os.path.join(base_of_BIDS_tree,'rawdata'))
	all_subjects_in_layout=layout.get_subjects()
	
	subjects_to_be_skipped_because_already_present=[]
	
	if skip_if_present:		#skip those subjects for whom all available sessions have already been MRIQC'ed. This assumes that sessions that used to be available in rawdata, remain available.
		
		num_sessions_to_be_analyzed_per_subject=[len(layout.get_sessions(subject=one_sub)) for one_sub in all_subjects_in_layout]
		
		all_subject_folders_in_output=[os.path.split(candidate)[1] for candidate in glob.glob(os.path.join(output_folder,'sub-*')) if os.path.isdir(candidate)]
		for sub_index,one_sub in enumerate(all_subjects_in_layout):
			if 'sub-'+one_sub in all_subject_folders_in_output:
				num_sessions_analyzed_this_subject=len(glob.glob(os.path.join(output_folder,'sub-'+one_sub,'ses-*')))
			else:
				num_sessions_analyzed_this_subject=0
		
			if num_sessions_analyzed_this_subject==num_sessions_to_be_analyzed_per_subject[sub_index]:
				fMRI_tools_logger.info('Skipping '+'sub-'+one_sub+' because MRIQC output is already present for a number of sessions that is equal to all sessions available in rawdata for this participant.')
				subjects_to_be_skipped_because_already_present+=[one_sub]
	
	if 	use_podman:
		
		shell_command_arguments=['podman','run','-it','--rm',\
			'-v',os.path.join(base_of_BIDS_tree,'rawdata')+':/data:ro',\
			'-v',output_folder+':/out',\
			'nipreps/mriqc:latest','/data','/out']	#these are the call to podman and its arguments
			
		shell_command_arguments+=[participant_or_group,\
			'-v']									#the -v here has to do with the verbosity of the MRIQC output; has nothing to do with the -v used in the context of mounting docker/podman volumes
		
		if participant_or_group=='participant':
		
			shell_command_arguments+=['--participant-label']
		
			if not(subjects):
			
				subjects=all_subjects_in_layout
			
			subjects_included=[one_sub for one_sub in subjects if not one_sub in subjects_to_be_skipped_because_already_present]
			shell_command_arguments+=subjects_included
		
		if len(subjects_included)==0 and participant_or_group=='participant':
			fMRI_tools_logger.info('On second thoughts: not running MRIQC at all because output is already present for all subjects and sessions that are available in rawdata.')
		else:
			if run_option=='now':
				general_tools.execute_shell_command(shell_command_arguments,collect_response=False,wait_to_finish=wait_to_finish)
			else:
				os.makedirs(os.path.join(output_folder,'scripts'), exist_ok=True)
			
				script_text_file_name_no_suffix='run_me_'+participant_or_group+'_'+'_'.join(subjects)
			
				if len(script_text_file_name_no_suffix)>max_text_file_name_length:
					suffix='_abbrev'
					script_text_file_name_no_suffix=script_text_file_name_no_suffix[:max_text_file_name_length-len(suffix)]+suffix
				script_text_file_name=script_text_file_name_no_suffix+'.txt'
				script_path=os.path.join(output_folder,'scripts',script_text_file_name)
				with open(script_path, 'w') as f:
					f.write('Copy the following line and paste it in a terminal to run MRIQC:')
					f.write(' '.join(shell_command_arguments))
		
	else:
		
		docker_command=' /data /out '+participant_or_group+' -v'		#the -v here has to do with the verbosity of the MRIQC output; has nothing to do with the -v used in the context of mounting docker volumes
	
		if participant_or_group=='participant':
		
			docker_command+=' --participant-label'
		
			if not(subjects):
			
				subjects=all_subjects_in_layout
		
			subjects_included=[one_sub for one_sub in subjects if not one_sub in subjects_to_be_skipped_because_already_present]
			docker_command+=' '
			docker_command+=' '.join(subjects_included)
	
		if len(subjects_included)==0 and participant_or_group=='participant':
			fMRI_tools_logger.info('On second thoughts: not running MRIQC at all because output is already present for all subjects and sessions that are available in rawdata.')
		else:
			client = docker.from_env()	#the client is the guy who you can say things like 'docker run' to, and who will then send those commands to the docker deamon, who interacts with the actual images/containers
			volumes = [os.path.join(base_of_BIDS_tree,'rawdata')+':/data:ro',output_folder+':/out']
			client.containers.run('nipreps/mriqc:latest',volumes=volumes,command=docker_command, detach=not(wait_to_finish))
			#this is equivalent to, in the terminal, running something like 'docker run -v '+os.path.join(base_of_BIDS_tree,'rawdata')+':/data:ro'+' -v '+output_folder+':/out nipreps/mriqc:latest '+docker_command
		

def shifted_spm_hrf(tr,derivative_weight,oversampling=50,time_length=32.0,onset=0.0):
	
	"""Compute a time-shifted version of the canonical SPM hemodynamic response function by adding a scaled version of the derivative to that function to it.
	
	Arguments (all arguments are the same as those of nilearn.glm.first_levelspm_hrf() and nilearn.glm.first_levelspm_time_derivative()
			except derivative_weight):
		tr (float):	Scan repeat time, in seconds.
		derivative_weight (float): weight of the derivative function that is added to the non-derivative (which has weight 1)
		oversampling (int, optional): Temporal oversampling factor. Default=50.
		time_length (float, optional): hrf kernel length, in seconds. Default=32.
		onset (float, optional): Onset of the response in seconds. Default=0.
	
	Returns (same type of data as nilearn.glm.first_levelspm_hrf() and nilearn.glm.first_levelspm_time_derivative()):
		shifted_hrf (numpy array of shape oversampling*time_length/tr, filled with floats): response values as a function of time
	
	"""
	
	shifted_hrf=spm_hrf(tr, oversampling, time_length, onset) + derivative_weight*spm_time_derivative(tr, oversampling, time_length, onset)
	return shifted_hrf


def shifted_spm_hrf_function_generator(derivative_weight = 1):
	
	"""Generate a specific function that computes a time-shifted version of the canonical SPM HRF. So a function like shifted_spm_hrf(), but a
		specific one in which derivative_weight has been fixed rather than being passed as an argument. This is necessary if the function is to
		be used as an HRF function in nilearn.glm.first_level.first_level_from_bids().
	
	Arguments:
		derivative_weight (float): weight of the derivative function that is added to the non-derivative (which has weight 1)
	
	Returns:
		_function (function that takes arguments tr, oversampling, time_length, onset): function that relates time to response strength based on
			shifted canonical SPM HRF.
	
	Example usage:
		derivative_weight=0.1
		shifted_HRF_function=fMRI_tools.shifted_spm_hrf_function_generator(derivative_weight)
		
		models, models_run_imgs, models_events, models_confounds = \
			first_level_from_bids(data_path_raw, task_label, space_label,
			derivatives_folder=data_path_input,
			t_r=repetition_time,
			noise_model='ar1',
			hrf_model=shifted_HRF_function,
			drift_model=my_drift_model,
			high_pass=my_high_pass,
			signal_scaling=0,
			minimize_memory=False,
			standardize=False)
	"""
	
	def hrf(tr,oversampling=50,time_length=32.0,onset=0.0):
		shifted_hrf=spm_hrf(tr, oversampling, time_length, onset) + derivative_weight*spm_time_derivative(tr, oversampling, time_length, onset)
		return shifted_hrf
	
	return hrf
	
	
def z_map_to_func_mask_nifti(z_map_path,z_threshold,dilation_iterations=None):
	
	"""Create a nifti object with 1s and 0s that specify a functionally defined region, based on the path of a nifti file that contains z-scores.
	
	Arguments:
		z_map_path (string): absolute path of z-score file
		z_threshold (float): any voxel with a z-score larger than that will get a 1
		dilation_iterations (int, optional): iterations of binary dilation, if we want to expand the mask a bit.
	
	Returns:
		z_map_thresholded_nifti (nifti object): in the same space as the nifti file at z_map_path, but filled with 0s and 1s.
	"""
	
	z_map_nifti=nib.load(z_map_path)
	z_data_thresholded = np.array(z_map_nifti.get_fdata()>z_threshold, dtype=np.int8)
	z_map_thresholded_nifti = nib.Nifti1Image(z_data_thresholded, z_map_nifti.affine, z_map_nifti.header)
	z_map_thresholded_nifti.set_data_dtype('i1') 
	
	if dilation_iterations:
		z_map_dilated = binary_dilation(z_map_thresholded_nifti.get_fdata(), iterations=dilation_iterations).astype(np.int8)
		z_map_thresholded_nifti=nib.Nifti1Image(z_map_dilated, z_map_thresholded_nifti.affine, z_map_thresholded_nifti.header)
	
	return z_map_thresholded_nifti

	
