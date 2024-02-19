"""
This code computes 1st level contrasts between face stimulus presentations and an implicit baseline, basing stats on the 5 repetitions within each observer.
Based in large part on https://peerherholz.github.io/workshop_weizmann/advanced/statistical_analyses_MRI.html
See also: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html
"""

from ast import And
from bids import BIDSLayout
from IPython import embed as shell
import json
from nilearn.glm.first_level import first_level_from_bids, make_first_level_design_matrix
from nilearn.image import resample_to_img, math_img, concat_imgs
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn._utils.glm import full_rank
import os
import numpy as np
import re
import nibabel as nib
from scipy.ndimage import binary_dilation
import copy
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import random
import matplotlib
import matplotlib.pyplot as pl
import sys
import pandas
from  statsmodels.stats.outliers_influence import variance_inflation_factor
#import seaborn as sns
import math

#Make available folders above this script's folder, so that we can import tools.[stuff]:
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# grandparent = os.path.dirname(parent)
# sys.path.append(grandparent)

import tools.general_tools as general_tools
import tools.BIDS_tools as BIDS_tools
import tools.fMRI_tools as fMRI_tools

base_of_BIDS_tree='/fmri/PI/thakkar/Face_Int/data_pipeline_Jan_starting_April_23'
main_script_folder_name='analysis_code_face_integration'
data_path_raw = os.path.join(base_of_BIDS_tree,'rawdata')					#the 'raw' data path in the BIDS structure. Used by nilearn.glm.first_level.first_level_model_from_bids to figure out stuff about data organization, but the images there do not form input to actual model.
data_path_input = os.path.join(base_of_BIDS_tree,'derivatives','fMRIPrep')	#where the actual input to the model is

task_label = 'main'
space_label = 'MNI152NLin2009cAsym'
nuisance_regressors_included=['white_matter', 'global_signal', 'framewise_displacement', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

my_hrf_model='spm + derivative'
my_drift_model='cosine'
my_high_pass=1./1000	#scans last about 280 s, and we're contrasting the first and last ~20 s to the rest. So we can't afford to use a cut-off that's faster than a few multiples of 1/280 Hz.

#------------
readme_text = "This derivatives folder contains nii.gz files with whole-brain patterns of z-stat patterns for the contrast between face image presentations and an implicit baseline. \
Because the baseline is formed by periods that nothing whatsoever happened, you should get activation patterns that \
are consistent not just with face processing but with visual processing, task engagement and motor action." #a description of what the derivatives folder contains

data_path_output=BIDS_tools.prepare_derivatives_folder(base_of_BIDS_tree, main_script_folder_name, readme_text)

#Query the BIDS data structure for some information:
layout = BIDSLayout(data_path_raw,derivatives=data_path_input)
relevant_json_filename=layout.get(extension=".json", task=task_label,return_type='filename', scope='derivatives', space='T1w', desc='preproc')[0]
repetition_time = json.load(open(relevant_json_filename))["RepetitionTime"]
start_time = json.load(open(relevant_json_filename))["StartTime"]

#Call first_level_from_bids to create a model object, a list of the runs,
#a list of events, and a list of confounds for each subject. The latter two of those (events and confounds) will have to be amended later
#in order to run a GLM to obtain the events we actually want to use in the GLM, and in order to get the confounds we'll actually want to use:
models, models_run_imgs, models_events, models_confounds = \
	first_level_from_bids(data_path_raw, task_label, space_label,
	derivatives_folder=data_path_input,
	t_r=repetition_time,
	noise_model='ar1',
	hrf_model=my_hrf_model,
	drift_model=my_drift_model,
	high_pass=my_high_pass,
	signal_scaling=0,
	minimize_memory=False,
	standardize=False,
	slice_time_ref = start_time/repetition_time
	)

# models = a list of FirstLevelModels, one for each subject
# models_run_imgs = a list of lists, one for each subject. Each subject specific list contains a list of paths to all of the main runs for that subject
# models_events = a list of lists, one for each subject. Each subject specific list contains a list of data frames containing the behavioral data, one df for each run that subject completed
# models_confounds = a list of lists, one for each subject. Each subject specific list contains a list of data frames containing the confound data, one df for each run that subject completed

# Whereas first_level_from_bids grabbed all events available in the json files, we now home in on the events we actually want to use in our GLM.
# The below code creates pairs of column names and value of interest (in reference to models_events that first_level_from_bids created), 
# paired with an event name (in reference to the new models_events_of_interest that we'll create). In other words it takes the column name "trial type" with value "face_image" and gives it the event name "face_image". 
# As it happens, in this case the event type is defined by a single column; in the commented-out code there are examples of other event types that are defined by multiple columns.

column_value_pairs_plus_name=[[{'trial_type':'face_image'},'face_image']]

# for image_index in range(num_images):
# 	column_value_pairs_plus_name+=[[{'trial_type':'mem_image','image_identity': image_index}, 'mem_image_' + str(image_index)], [{'trial_type':'test_image','image_identity': image_index, 'cued_image_or_no': 0}, 'uncued_test_image_' + str(image_index)], [{'trial_type':'test_image','image_identity': image_index, 'cued_image_or_no': 1}, 'cued_test_image_' + str(image_index)]]

column_value_pairs=[entry[0] for entry in column_value_pairs_plus_name] # just the column val pairs, same length as above
derived_event_names=[entry[1] for entry in column_value_pairs_plus_name] # just the derived event names, same length as above

models_events_of_interest=fMRI_tools.build_models_events(models_events,column_value_pairs,derived_event_names) #creates a new version of models_events which only contains events we care about

models_confounds_no_nan=fMRI_tools.build_models_confounds(models_confounds,nuisance_regressors_included)	#first_level_from_bids also grabbed all confounds that are possible. So we need to reduce to those we want (and also fix some NaN thing)

model_and_args = zip(models, models_run_imgs, models_events_of_interest, models_confounds_no_nan)

for model_index, (model, imgs, events, confounds) in enumerate(model_and_args):	#loop over models (so over participants)
	
	this_sub=re.search("sub-(.*?)_", os.path.split(imgs[0])[-1])[1]

	events_original=copy.deepcopy(events)

	output_functional_dir_this_sub=os.path.join(data_path_output,'sub-'+this_sub,'func')
	os.makedirs(output_functional_dir_this_sub, exist_ok=True)
	
	frame_times_per_run = [np.array([start_time+repetition_time*slice_index for slice_index in range(len(confounds_one_obs))]) for confounds_one_obs in models_confounds[model_index]]	#get the times in seconds for all fMRI samples within each run. Using 'models_confounds' for this rather than 'confounds' so that it still works if no confounds are included in nuisance_regressors_included.
	
	design_matrices = []
	
	for run_index,events_this_run in enumerate(events_original):	#loop over the experimental runs
		
		design_matrix_this_run=make_first_level_design_matrix(frame_times_per_run[run_index],events=events[run_index], hrf_model=model.hrf_model, drift_model=model.drift_model, high_pass=model.high_pass, drift_order=model.drift_order,add_regs=confounds[run_index])
		design_matrices.append(design_matrix_this_run)

	fmri_glm = model.fit(imgs, design_matrices=design_matrices)
	
	#Save a plot of an example design matrix:
	design_matrix_for_plotting=model.design_matrices_[0]
	desc_value=re.search("desc-([a-zA-Z0-9]*)",data_path_output)[1]		#this takes whatever combination of letters and numbers follows 'desc-' in data_path_output
	
	entities = {
		'sub': this_sub,
		'task': task_label,
		'desc': desc_value,
		'run': '0',
		'suffix': 'designMatrix',
		'extension': '.png'
	}
	design_matrix_file=BIDS_tools.my_build_bids_filename(entities)

	full_path_design_matrix=os.path.join(data_path_output,'sub-'+this_sub,'figures',design_matrix_file)
	os.makedirs(os.path.split(full_path_design_matrix)[0], exist_ok=True)

	plot_design_matrix(design_matrix_for_plotting, output_file=full_path_design_matrix)
	
	#to extract the per-voxel z-stats of one event type versus baseline, we create a contrast definition with a weight of 1 for that event type (i.e. regressor) and 0s for everyone else,
	#and then we compute the contrast with 'z_score' as the output_type:
	contrast_definition_across_runs=[]
	for design_matrix_one_run in model.design_matrices_:
		design_matrix_column_names=design_matrix_one_run.keys()
		this_contrast=np.zeros(len(design_matrix_column_names))
		regressor_index_face_images=np.where(design_matrix_column_names.str.contains('face_image') & ~(design_matrix_column_names.str.contains('derivative')))[0]
		this_contrast[regressor_index_face_images]=1		
		contrast_definition_across_runs+=[this_contrast]
	
	#Save a plot of an example contrast vector:
	entities['suffix']='contrastMatrix'
	contrast_file=BIDS_tools.my_build_bids_filename(entities)
	
	full_path_contrast_matrix=os.path.join(data_path_output,'sub-'+this_sub,'figures',contrast_file)
	
	plot_contrast_matrix(contrast_definition_across_runs[0], design_matrix=model.design_matrices_[0], output_file=full_path_contrast_matrix)
	
	z_stat_map = model.compute_contrast(contrast_definition_across_runs,output_type='z_score')
	
	del entities['run']
	entities['space']=space_label
	entities['suffix']='zStats'
	entities['extension']='.nii.gz'

	z_stat_filename=BIDS_tools.my_build_bids_filename(entities)

	full_path_z_stat_file=os.path.join(data_path_output,'sub-'+this_sub,'func',z_stat_filename)
	
	z_stat_map.to_filename(full_path_z_stat_file)
	
	#Open the z-stat data in MRIcroGL or generate script to do so later:
	anatomical_path=BIDS_tools.get_anatomical_path(layout,this_sub,space=space_label)
	
	thresholds=[2.3, 3.1]
	fMRI_tools.open_in_MRIcroGL(anatomical_path,overlay_info=[[full_path_z_stat_file,'1red',thresholds],[full_path_z_stat_file,'3blue',[-value for value in thresholds]]],\
	short_identifier=desc_value,run_option='later',mountpoint_local_remote=['/Users/janbrascamp/mountpoint_thakkar','/fmri/PI/thakkar/Face_Int'])	#Open z map on top of anatomical in MRIcroGL
