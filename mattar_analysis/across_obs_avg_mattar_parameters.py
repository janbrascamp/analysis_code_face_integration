# ------ File description ------
"""
This code takes individual observers' output of run_mattar_analysis_on_MSU_face_data.m, and averages the R^2 data (sub-[...]_task-main_space-MNI152NLin2009cAsym_desc-mattarAdapt_R2.nii),
the adaptMu data (sub-[...]_task-main_space-MNI152NLin2009cAsym_desc-_mattarAdapt_adaptMu.nii), and the adaptGain data (sub-[...]_task-main_space-MNI152NLin2009cAsym_desc-_mattarAdapt_adaptGain.nii)
across observers.

To run: in an SSH terminal to the Circ server, activate the right conda environment using 'conda activate fMRI', and then run it like this: py across_obs_avg_mattar_parameters.py
"""

# ------ Imports ------
from IPython import embed as shell
import os
import glob
import sys
import numpy
from nilearn.image import math_img, mean_img
from nilearn.surface import vol_to_surf
from nilearn.datasets import fetch_surf_fsaverage
import nibabel as nib
import re

import sys
sys.path.append('..')
import tools.fMRI_tools as fMRI_tools

# ------ Adjustable parameters -------

r2_threshold=.2
desc_strings=["mattarAdapt_R2","_mattarAdapt_adaptGain","_mattarAdapt_adaptMu"]	#the strings, in the 'desc' entity of the file name, that identify the parameters we're interested in
subject_search_string='sub-C*'			#(if you don't identify specific subjects below, then) this string determines which category of subjects to include in the average. All controls start with 'sub-C'; all patients with 'sub-A'; all subjects with 'sub-'
output_folder='mattar_analysis_across_obs'	#name of output folder within BIDS data structure

# ------ Set up variables to point the code to the data locations, and determine the output path ------
input_data_path = '/array/fmri/PI/thakkar/Face_Int/data_pipeline_Jan_starting_April_23/derivatives/mattar_analysis'
session_number = 1

output_derivatives_path=os.path.join(input_data_path.split('derivatives/')[0],'derivatives',output_folder)

"""
Select subjects: either define some or, alternatively, automatically include everyone in input_data_path.
"""
subjects=[]	#enter subjects here if you want to analyze a subset. Format: ['sub-A0001','sub-A0007']
if not subjects:
	subjects=glob.glob1(input_data_path,subject_search_string)

"""
For each of the parameters / variables that you'd like to see an average for, compute the average.
"""
for desc_string in desc_strings:
	
	"""
	For each subject, get the absolute path of the 3D nifti that contributes to the average.
	"""
	
	all_absolute_nift_paths=[]
	for subject in subjects:
	
		input_data_path_this_sub=os.path.join(input_data_path,subject,'ses-'+str(session_number),'func')
		
		file_name=glob.glob1(input_data_path_this_sub,'*desc-'+desc_string+'.nii')[0]
		
		all_absolute_nift_paths+=[os.path.join(input_data_path_this_sub,file_name)]
	
	"""
	Get some info from one of the filenames just collected, to inform the across-participant average filename and folder placement.
	"""
	
	one_input_filename=os.path.split(all_absolute_nift_paths[0])[1]
	task=re.findall('task-([a-zA-Z0-9]*)_',one_input_filename)[0]
	space=re.findall('space-([a-zA-Z0-9]*)_',one_input_filename)[0]
	
	output_folder=os.path.join(output_derivatives_path,'task-'+task+'_space-'+space,'func')
	os.makedirs(output_folder,exist_ok=True)
	
	"""
	Compute the average and save it to file (creating the required folder structure in the process).
	"""
	
	average_image=mean_img(all_absolute_nift_paths)
	output_filename='task-'+task+'_space-'+space+'_desc-'+desc_string+'.'+one_input_filename.split('.')[-1]
	nib.save(average_image, os.path.join(output_folder,output_filename))
	
	fsaverage = fetch_surf_fsaverage()
	
	for hemisphere_index,hemisphere_string in enumerate(['rh','lh']):
		output_filename='task-'+task+'_space-fsaverage5_desc-'+desc_string+'_'+hemisphere_string+'.gii'
		surface_data = vol_to_surf(average_image, [fsaverage.pial_right,fsaverage.pial_left][hemisphere_index])
		surface_img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(surface_data)])
		nib.save(surface_img,os.path.join(output_folder,output_filename))

"""
Also produce a version of the across-observer average mu data that is masked based on a threshold applied to the across-observer average r^2 value.
"""
r2_filename=glob.glob1(output_folder,'*_R2.nii')[0]
r2_threshold_nifti=fMRI_tools.z_map_to_func_mask_nifti(os.path.join(output_folder,r2_filename),r2_threshold)

mu_filename=glob.glob1(output_folder,'*adaptMu.nii')[0]
mu_nifti=nib.load(os.path.join(output_folder,mu_filename))
mu_nifti_masked=math_img('mu_img * mask_img',mu_img=mu_nifti,mask_img=r2_threshold_nifti)	#mask the mu data based on the thresholded r^2 data

mu_filename_split=mu_filename.split('.')
output_filename_volume='.'.join([mu_filename_split[0]+'_r2GT'+str(int(r2_threshold*100))+'percent',mu_filename_split[1]])
nib.save(mu_nifti_masked, os.path.join(output_folder,output_filename_volume))
	
for hemisphere_index,hemisphere_string in enumerate(['rh','lh']):
	output_filename=output_filename_volume.split('.')[0]+'_'+hemisphere_string+'.gii'
	
	surface_data_mu = vol_to_surf(mu_nifti, [fsaverage.pial_right,fsaverage.pial_left][hemisphere_index])
	surface_data_r2 = vol_to_surf(nib.load(os.path.join(output_folder,r2_filename)), [fsaverage.pial_right,fsaverage.pial_left][hemisphere_index])
	
	surface_data_mu_masked=surface_data_mu*(surface_data_r2>r2_threshold)
	
	surface_img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(surface_data_mu_masked)])
	nib.save(surface_img,os.path.join(output_folder,output_filename))

