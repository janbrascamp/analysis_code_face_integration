# ------ File description ------
"""
This code grabs behavioral data and scanner data from temporary holding areas, moves those data to the right places, and performs preprocessing on those data.
After having run this code, the data are ready for actual analyses.

#The BIDS-compliant organization we use involves a directory tree that includes, right at its base, includes 3 folders:
#1. 'sourcedata'. This is where data are moved from the holding directories that temporarily hold newly-acquired data. Minimal or no processing takes place in
	this moving step. After this moving step all unprocessed data pertaining to a given scanning session are permanently
	stored in 'sourcedata', and holding directories can be emptied out whenever. 'sourcedata' itself is not at all BIDS-compliant: the organization
	(i.e. directory structure, file naming conventions, and some other things) of this folder is much closer to the way things looked in the holding directories
	than the way things are supposed to look according to BIDS. If the variable 'subjects' below is defined on the basis of behavior_scan_pairing_info_path, then it is critical that
	the /behavior subfolder inside sourcedata has one folder for each scanning session, and that that folder follows naming conventions. Specificially:
	1. the folder name must have the subject's behavioral ID in it as a unique identifier, in correspondence with the leftmost column of behavior_scan_pairing_info_path.
	2. If the subject performed multiple scans, then it is critical that the folder naming for that participant is such, that alphabetical sorting will place the folders
		in order of ascending scan number, unless the argument 'sort_behav_session_order manually' is set to True when calling BIDS_tools.fill_rawdata_folder.
#'rawdata'. This folder is fully BIDS-compliant. This folder contains effectively the same data files as sourcedata, but check out the subfolder structure and file naming
	conventions, as well as the choice of file types, including the .json files (which you can open in a text editor) that provide background information on each of
	the data files in here. That's one of the great strengths of BIDS: it's easy to navigate these files and not get lost, and it's easy for anyone else who knows the
	BIDS standard to instantly know what's what.
#'derivatives'. This folder adheres quite closely to the BIDS standard, although it lacks a few things that would make it fully BIDS-compliant. Each analysis we perform,
	unless otherwise specified, takes its data from the 'rawdata' folder and puts its results in a subfolder of 'derivatives'. So each subfolder of 'derivatives'
	contains the results of one analysis. That way things stay organized, even in the face of many pilot analyses and side analyses.

The current code right here takes data from the holding areas, and uses it to populate first the 'sourcedata' folder and then the 'rawdata' folder. Then this code
runs a few quality control and preprocessing analyses on the data in 'rawdata', and puts the result of each of those analyses in its own subfolder of the 'derivatives' folder.
"""

# ------ Imports ------
from IPython import embed as shell
import os
import glob
import tools.BIDS_tools as BIDS_tools
import tools.fMRI_tools as fMRI_tools
import tools.general_tools as general_tools
import tools.ad_hoc_tools as ad_hoc_tools

# ------ Set up paths to data to be used ------
# easily update experiment name here instead of in multiple spots
experiment_name = 'Face_Int'
experiment_name_text = 'Thakkar lab face integration project'
holding_dir_mri='/fmri/holding'			#All scanner files go here initially because John Irwin uploads them there. Our own data are there and others', too. Data may be removed from here periodically I suppose.
raw_data_identifying_string='thakkar*'		#Which string (including wildcards) identifies relevant data in the holding_dir_mri? There's a ton of data in there from a lot of people but we're the only ones whose files start with 'bra'
base_of_BIDS_tree=os.path.join('/array/fmri/PI/thakkar',experiment_name, 'data_pipeline_Jan_starting_April_23')	#This variable points to the basis of the directory tree we'll use. It is the folder within which we can find 'sourcedata', 'rawdata', and 'derivatives'
holding_dir_behavior=os.path.join(base_of_BIDS_tree,'_fMRI_Behavioral_Output')	#You need to manually move to-be-included behavioral files here before running this code.
behavior_scan_pairing_info_path='/home/brascamp/analysis_code_face_integration/behavior_scanner_ID_pairing.csv'	#Points to a .tsv file that indicates which behavioral subject name (left column) corresponds with which scanner folder(s)
												#(all columns to the right of the leftmost one). Those scanner folders should be provided in the order (columns going right) in which you want the sessions to appear in the folder structure.
scan_runs_to_skip_info_path='/home/brascamp/analysis_code_face_integration/erroneous_scanner_runs.csv'	#Points to a .tsv file that indicates, for specific scan sessions, which MRI scan numbers should not be transferred from the 
#sourcedata folder to the rawdata folder. Leftmost column is the scan folder, all columns to the right of that are runs that should be skipped.

# ------ Set global variables ------
#subjects=['103']		#Here you can list behavioral identifiers of participants you want to run. If you want to run everyone, then do not define 'subjects' at all, and make sure behavior_scan_pairing_info_path is defined. Then all subjects listed in the 
						#first column of that file will be analyzed.
# we can set subjects to just the subjects we wish to include so that we do not rerun analyses
scan_types=[{'suffix':'T1w','series_description':'mprage'}, {'suffix':'T2w','series_description':'Inplane-highres'}, {'suffix':'bold','series_description':'fMRI','task':'main','example_events_json_file':'main_example_events.json'}]
# What types of scans may be found in the MRI subfolder of sourcedata. 
# 'suffix' and 'task' refer to the BIDS entities that should be used in the rawdata folder;
# 'series_description' is an indicator that uniquely defines that scan type in the
# 'review.out' files in sourcedata. The latter can be omitted if unsure, in which case choose it
# interactively later on.
# *** note: if you do not want to generate json files accompanying events, then do not include key/value pairs like 'example_events_json_file':'main_example_events.json'.
dataset_description={'Name': experiment_name_text,\
  'BIDSVersion': 'v1.6.0',
  'DatasetType': 'raw',
  'Authors': ['Jan Brascamp'],
  'Funding': ['National Institutes of Health R01MH121417'],
  'EthicsApprovals': ['Michigan State University IRB office Protocol STUDY00005053']}
#This text goes into the dataset_description json file at the top of the rawdata folder, required by BIDS.

readme_text="This is dataset collected in the context of the " + experiment_name_text + \
	"\nIn 'sourcedata' the folder 'behavior' contains behavioral data separated per scanning session." \
	"\nIn 'sourcedata' the folder 'MRI' contains scanner data separated per scanning session." \
	"\nIn 'sourcedata' the folder 'other_materials', if present, may contain other relevant information such as, for instance, image or sound files used for the experiment, experiment scripts, etc. Those materials may well have been moved there manually."

if 'holding_dir_code' in globals():
	readme_text+="\nIn 'sourcedata' the folder 'code' contains code used during individual scanning sessions."\
	
readme_text+="\nIn the README file in 'rawdata' you can find how the data there relate to the data in 'sourcedata'."
# This is the text that goes into the README file inside the 'sourcedata folder. It will also form the basis of the README file (required by BIDS) that goes inside the rawdata folder.

""" 0. Manually move behavioral folders and, optionally, eyetracking and code folders to their designated spots in holding
	* Each step below gives a summary of what is completed at each step
	* You can double check that the outputs are created accordingly along the way
"""

""" 1. Populate 'sourcedata'
	* Get data from holding areas to populate the 'sourcedata' folder
"""
if 'holding_dir_code' in globals():
	BIDS_tools.fill_sourcedata_folder(base_of_BIDS_tree,holding_dir_info_mri=[holding_dir_mri],raw_data_identifying_string_mri=raw_data_identifying_string,holding_dir_info_behavior=[holding_dir_behavior],holding_dir_info_code=[holding_dir_code],readme_text=readme_text)
else:
	BIDS_tools.fill_sourcedata_folder(base_of_BIDS_tree,holding_dir_info_mri=[holding_dir_mri],raw_data_identifying_string_mri=raw_data_identifying_string,holding_dir_info_behavior=[holding_dir_behavior],readme_text=readme_text)

""" 2. Unpack the scanner data and put the behavioral data in the right format
	* The scanner data come from the holding area in a compressed form (one package per session) and we need to unpack the files to work with them (split each session into runs)
	* The behavioral data come from the holding area in a form that is not BIDS compliant and we need to convert those data to work with them.
"""

general_tools.unpack_tar_gz_archives(os.path.join(base_of_BIDS_tree,'sourcedata','MRI'))
ad_hoc_tools.convert_behavioral_data_to_bids_tsv_for_face_project(os.path.join(base_of_BIDS_tree,'sourcedata','behavior'),face_position_file_path_sz='/home/brascamp/analysis_code_face_integration/xyz_coordinates_young_SZ.csv',face_position_file_path_hc='/home/brascamp/analysis_code_face_integration/xyz_coordinates_young_HC.csv',scan_folder_identifier_string='block*')


""" 3. Create template _events.json files
	If you want .json files to accompany tsv files for events in rawdata, then create or copy (a) relevant template json file(s) (one for each functional task type that you distinguish in the BIDS structure -- that distinction should be visible in scan_types above).
	Name such template files in correspondence with the value of key 'example_events_json_file' in scan_types above, and put it/them in sourcedata/behavior.
	This means that if you want make sure that a key/value pair(s) like 'example_events_json_file':'main_example_events.json' is/are present in the variable scan_types.
	If you do not want to create '_events.json' files, there is no need to create or copy example files. However, then you must remove any such key/value pairs from scan_types.
"""				

""" 4. Populate 'rawdata'
	* Determine the list of series numbers "Ser" from the now unpacked MRI data and the scan numbers "scan" from the behavioral data
	* Get data from 'sourcedata' to populate the 'rawdata' folder
	* Requires you to manually match subject and session data
"""
if not 'subjects' in globals():

	if behavior_scan_pairing_info_path:
		with open(behavior_scan_pairing_info_path) as f:
			behavior_scan_pairing_info=f.read()
		subjects=[info_one_row.split(',')[0] for info_one_row in behavior_scan_pairing_info.split('\n')]	#if no subjects preselected, then choose all that are listed in this file (but BIDS_tools.fill_rawdata_folder can automatically skip those subjects for whom data are already present in rawdata, depending on its arguments).
	else:
		raise Exception('You need to somehow identify the behavioral subject IDs to include!')

run_number_defining_re_behavior='block([0-9]*)_'
run_number_defining_re_MRI='Ser([0-9]*)_'
BIDS_tools.fill_rawdata_folder(base_of_BIDS_tree,subjects,scan_types,dataset_description,run_number_defining_re_MRI,run_number_defining_re_behavior,behavior_scan_pairing_info_path=behavior_scan_pairing_info_path,scan_runs_to_skip_info_path=scan_runs_to_skip_info_path,sort_behav_sessions_manually=False)

""" 4. Run quality control analysis on scanner data using MRIQC
"""
fMRI_tools.run_MRIQC(base_of_BIDS_tree,participant_or_group='participant',output_subfolder='MRIQC',wait_to_finish=False,use_podman=True,run_option='later')

""" 5. Run preprocessing on scanner data using fMRIPrep
"""
fMRI_tools.run_fMRIPrep(base_of_BIDS_tree,output_subfolder='fMRIPrep',path_to_freesurfer_license='/usr/local/freesurfer/7.3.2-1/license.txt',output_spaces=['MNI152NLin2009cAsym','T1w','fsaverage5'],wait_to_finish=True,use_podman=True,use_nohup=True,run_option='later')

# """ 6. If the run_option parameter in step 4 is set to 'later':
# 	* In the current experiment folder, open derivatives/MRIQC/scripts
# 	* You should see a text file which says "run_me" along with all participant numbers that you included in this run of holding_to_fmri_prepped (if this is not the first participant that you have run holding_to_fmnri_prepped for, make sure to select the file with the current participants in the name)
# 	* Open the text file. Copy the line of text beginning with "podman" (after "MRIQC:")
# 	* Open a terminal window on the server via x2go (important - this code takes a long time to run) and type "conda activate fMRI"
# 	* Paste the copied line into the terminal window and run
# 	* if prompted to select an "image", select the "docker" option
# 		Please select an image:
# 		 	registry.fedoraproject.org/nipreps/mriqc:latest
# 			registry.access.redhat.com/nipreps/mriqc:latest
# 			registry.centos.org/nipreps/mriqc:latest
# 		▸	docker.io/nipreps/mriqc:latest
# """
#
# """ 7. If the run_option parameter in step 5 is set to 'later':
# 	* In the current experiment folder, open derivatives/fMRIPrep/scripts
# 	* You should see a text file which says "run_me" along with all participant numbers that you included in this run of holding_to_fmri_prepped (if this is not the first participant that you have run holding_to_fmnri_prepped for, make sure to select the file with the current participants in the name)
# 	* Open the text file. Copy the line of text beginning with "podman" (after "fMRIPrep:")
# 	* Open a terminal window on the server via x2go (important - this code takes a long time to run) and type "conda activate fMRI"
# 	* Paste the copied line into the terminal window and run
# 	* if prompted to select an "image", select the "docker" option
# 		Please select an image:
# 		 	registry.fedoraproject.org/nipreps/mriqc:latest
# 			registry.access.redhat.com/nipreps/mriqc:latest
# 			registry.centos.org/nipreps/mriqc:latest
# 		▸	docker.io/nipreps/mriqc:latest
# """
#
#


