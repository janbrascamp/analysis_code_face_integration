# ------ File description ------
"""
This code takes individual observers' output of run_mattar_analysis_on_MSU_face_data.m, and applies a threshold to the observer's R^2 data (sub-[...]_task-main_space-MNI152NLin2009cAsym_desc-mattarAdapt_R2.nii),
to create a mask for the observer's adaptation parameter data (sub-[...]_task-main_space-MNI152NLin2009cAsym_desc-_mattarAdapt_adaptMu.nii and sub-[...]_task-main_space-MNI152NLin2009cAsym_desc-_mattarAdapt_adaptGain.nii).
It outputs files with the masked parameter values in the same folder(s) where it found the original run_mattar_analysis_on_MSU_face_data.m output.

To run: in an SSH terminal to the Circ server, activate the right conda environment using 'conda activate fMRI', and then run it like this: py mask_mattar_parameters.py
"""

# ------ Imports ------
from IPython import embed as shell
import os
import glob
import sys
import numpy
from nilearn.image import math_img
import nibabel as nib

import sys
sys.path.append('..')
import tools.fMRI_tools as fMRI_tools

# ------ Adjustable parameters -------
r2_thresholds=[.2,.4]

# ------ Set up variables to point the code to the data locations ------
mattar_data_path = '/array/fmri/PI/thakkar/Face_Int/data_pipeline_Jan_starting_April_23/derivatives/mattar_analysis'
session_number = 1

"""
Select subjects: either define some or, alternatively, automatically include everyone in mattar_data_path.
"""
subjects=[]	#enter subjects here if you want to analyze a subset. Format: ['sub-A0001','sub-A0007']
if not subjects:
	subjects=glob.glob1(mattar_data_path,'sub-*')
	
"""
For each subject, get the r^2 data. Then, for each value of r2_threshold, create a mask based on that threshold, and compute and output the masked adaptation parameter data using that mask.
"""
for subject in subjects:
	
	mattar_data_path_this_sub=os.path.join(mattar_data_path,subject,'ses-'+str(session_number),'func')
	
	r2_file_name=glob.glob1(mattar_data_path_this_sub,'*R2.nii')[0]
	
	for r2_threshold in r2_thresholds:
	
		r2_threshold_nifti=fMRI_tools.z_map_to_func_mask_nifti(os.path.join(mattar_data_path_this_sub,r2_file_name),r2_threshold)
	
		for adapt_parameter in ['adaptMu','adaptGain']:													#loop over the two adaptation parameters of interest
	
			adapt_parameter_file_name=glob.glob1(mattar_data_path_this_sub,'*'+adapt_parameter+'.nii')[0]
			adapt_parameter_nifti=nib.load(os.path.join(mattar_data_path_this_sub,adapt_parameter_file_name))		#load the parameter data
		
			adapt_parameter_nifti_masked=math_img('param_img * mask_img',param_img=adapt_parameter_nifti,mask_img=r2_threshold_nifti)	#mask the parameter data based on the thresholded r^2 data
		
			adapt_parameter_file_name_split=adapt_parameter_file_name.split('.')
			output_file_name='.'.join([adapt_parameter_file_name_split[0]+'_r2GT'+str(int(r2_threshold*100))+'percent',adapt_parameter_file_name_split[1]])
		
			nib.save(adapt_parameter_nifti_masked, os.path.join(mattar_data_path_this_sub,output_file_name))
