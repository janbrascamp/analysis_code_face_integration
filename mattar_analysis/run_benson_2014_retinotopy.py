# ------ File description ------
"""
This code uses anatomy to assign visual cortical area labels (plus other retinopy-related information) to the voxels in individual observers' brains, 
using neuropythy by Noah Benson (https://github.com/noahbenson/neuropythy/wiki/Retinotopy#prediction).

The surface-based data that form the basis of the analysis are taken from the sourcedata folder inside fMRIPrep's derivatives folder, where fMRIPrep puts its Freesurfer-related stuff.
The results are written to the same folder, using the same conventions as used by the neuropythy package's commands/benson14_retinotopy.py command

To run: in an SSH terminal to the Circ server, activate the right conda environment using 'conda activate fMRI', and then run it like this: py run_benson_2014_retinotopy.py
"""

# ------ Imports ------
from IPython import embed as shell
import os
import glob
import sys
import numpy
import neuropythy

# ------ Set up paths to data to be used ------
freesurfer_path = '/array/fmri/PI/thakkar/Face_Int/data_pipeline_Jan_starting_April_23/derivatives/fMRIPrep/sourcedata/freesurfer/'	#we base the analysis on the data in MRIPrep's sourcedata folder where it keeps Freesurfer data.

"""
Select subjects: either define some or, alternatively, automatically include everyone in freesurfer_path.
"""
subjects=[]	#enter subjects here if you want to analyze a subset. Format: ['sub-A0001','sub-A0007']
if not subjects:
	subjects=[element for element in glob.glob1(freesurfer_path,'*') if not 'fsaverage' in element]
	
"""
For each subject, predict retinotopy using neuropythy functionality, and save the result to disk.
"""
for subject in subjects:
	
	neuropythy_sub_object = neuropythy.freesurfer_subject(os.path.join(freesurfer_path,subject))	#identify the Freesurfer subject path and put it in the right variable type
	(lh_retino, rh_retino) = neuropythy.vision.predict_retinotopy(neuropythy_sub_object)			#get the retinotopy predictions
	
	for key in ['angle','eccen','sigma','varea']:													#loop over the various pieces of relevant information returned by predict_retinotopy
		
		for the_hemisphere_index in [0,1]:															#loop over the hemispheres to store the surface-based data
			hemisphere_data=[lh_retino,rh_retino][the_hemisphere_index]
			hemisphere_string=['lh','rh'][the_hemisphere_index]
		
			neuropythy.save(os.path.join(freesurfer_path,subject,'surf',hemisphere_string+'.benson14_'+key),hemisphere_data[key],format='freesurfer_morph')
			
		#and store the volume-based data for both hemispheres together. This part borrows heavily from the neuropythy package's commands/benson14_retinotopy.py command
		dtyp = (numpy.int32 if key == 'varea' else numpy.float32)		
		im = neuropythy_sub_object.images['brain']
		addr = (neuropythy_sub_object.lh.image_address(im), neuropythy_sub_object.rh.image_address(im))
		vol = neuropythy_sub_object.cortex_to_image((lh_retino[key], rh_retino[key]), neuropythy.mri.image_clear(im),method=('nearest' if key == 'varea' else 'linear'),address=addr, dtype=dtyp)
		
		neuropythy.save(os.path.join(freesurfer_path,subject,'mri', 'benson14_'+key+'.mgz'), vol)
