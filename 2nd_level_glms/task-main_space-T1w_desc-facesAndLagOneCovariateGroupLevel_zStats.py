"""
This code computes across-observer contrasts between face stimulus presentations and an implicit baseline, using the output of task-main_space-T1w_desc-facesVsBaseline_zStats.py as input.
Based in large part on https://peerherholz.github.io/workshop_weizmann/advanced/statistical_analyses_MRI.html
"""

from ast import And
from bids import BIDSLayout
from IPython import embed as shell
import json
from nilearn.glm.first_level import first_level_from_bids, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import resample_to_img, math_img, concat_imgs
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn._utils.glm import full_rank
from nilearn.datasets import load_mni152_template
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
data_path_input = os.path.join(base_of_BIDS_tree,'derivatives','1st_level_glms','task-main_space-T1w_desc-facesAndLagOneCovariate_zStats')	#where the first-level model results are
first_level_contrast_names = ['task-main_space-MNI152NLin2009cAsym_desc-facesAndLagOneCovariateFaceImage_zStats.nii.gz','task-main_space-MNI152NLin2009cAsym_desc-facesAndLagOneCovariateLagOneDistance_zStats.nii.gz']
full_path_mni_template=os.path.join(base_of_BIDS_tree,'sourcedata','fsl','data','mni152_template_1mm.nii.gz')

#------------
readme_text = "This derivatives folder contains across-observer whole-brain results for the contrasts that were computed in a GLM that includes \
both a face image regressor and a lag-one distance covariate. At the within-observer level this GLM is part of the computations in "+os.path.split(data_path_input)[-1]+".py" #a description of what the derivatives folder contains

data_path_output=BIDS_tools.prepare_derivatives_folder(base_of_BIDS_tree, main_script_folder_name, readme_text)

#Query the BIDS data structure for some information:
layout = BIDSLayout(data_path_raw,derivatives=data_path_input)

for first_level_contrast_name in first_level_contrast_names:
	
	#Get the first level z stat files:
	first_level_contrast_entities=BIDS_tools.my_get_bids_entities(first_level_contrast_name)
	list_of_z_maps=[candidate_filename for candidate_filename in layout.get(scope='derivatives', return_type='file', task=first_level_contrast_entities['task'], suffix=first_level_contrast_entities['suffix'], extension=['nii', 'nii.gz']) if re.search("_space-"+first_level_contrast_entities['space'], candidate_filename) and re.search("_desc-"+first_level_contrast_entities['desc'], candidate_filename)]
	#design matrix is just a bunch of ones (one for each observer):
	design_matrix = pandas.DataFrame([1] * len(list_of_z_maps),columns=['intercept'])

	second_level_model = SecondLevelModel()
	second_level_model = second_level_model.fit(list_of_z_maps, design_matrix=design_matrix)

	z_stat_map_group = second_level_model.compute_contrast(output_type='z_score')

	second_level_z_stat_file_name = first_level_contrast_name

	output_functional_dir=os.path.join(data_path_output,'func')
	os.makedirs(output_functional_dir, exist_ok=True)

	full_path_group_level_z_stat_file=os.path.join(output_functional_dir,second_level_z_stat_file_name)
	z_stat_map_group.to_filename(full_path_group_level_z_stat_file)

	#Open the z-stat data in MRIcroGL or generate script to do so later:

	if not(os.path.isfile(full_path_mni_template)):		#we need to open it on an average MNI brain so we download that if it doesn't exist yet
	
		location_mni_template=os.path.split(full_path_mni_template)[0]
		os.makedirs(location_mni_template, exist_ok=True)
	
		anatomical_nifti=load_mni152_template()
		anatomical_nifti.to_filename(full_path_mni_template)

	desc_value=first_level_contrast_entities['desc']

	thresholds=[2.3, 3.1]
	fMRI_tools.open_in_MRIcroGL(full_path_mni_template,overlay_info=[[full_path_group_level_z_stat_file,'1red',thresholds],[full_path_group_level_z_stat_file,'3blue',[-value for value in thresholds]]],\
	short_identifier=desc_value,run_option='later',mountpoint_local_remote=['/Users/janbrascamp/mountpoint_thakkar','/fmri/PI/thakkar/Face_Int'])	#Open z map on top of anatomical in MRIcroGL
