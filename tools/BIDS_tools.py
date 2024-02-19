from socket import if_nametoindex
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
import logging
import inspect
from datetime import datetime

from . import general_tools, fMRI_tools

BIDS_tools_logger = general_tools.get_logger(__name__)

def add_postprocessing_to_existing_derivatives_folder(derivatives_path):
	"""This function is to be used when adding postprocessing results to an existing derivatives folder. The function writes to the README file to state that the calling
		script has been writing postprocessing results to the folder, and the function also creates a designated 'postprocessing' folder inside the derivatives folder
		and returns its path.
	
	Arguments:
		derivatives_path (string). Absolute path of the derivatives folder to which the results of postprocessing should be written.
	
	Returns:
		postprocessing_path (string: Absolute path of the postprocessing folder inside full_derivatives_path, to which postprocessing results can be written
	"""
	
	BIDS_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	calling_script_full_path=re.search(".*file \'(.*)\'",str(inspect.stack()[1][0]))[1]
	
	postprocessing_path=os.path.join(derivatives_path,'postprocessing')
	os.makedirs(postprocessing_path, exist_ok=True)
	
	"""each time this function is called by a script, add mention of that to the README file inside full_derivatives_path"""
	now = datetime.now()
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
	readme_file = open(os.path.join(derivatives_path,'README'),"a")
	readme_file.write('\n\nA postprocessing script named '+calling_script_full_path+' wrote to this derivatives folder at the following time: '+date_time)
	readme_file.close()	
	
	return postprocessing_path
	

def fill_rawdata_folder(base_of_BIDS_tree,subjects,scan_types,dataset_description='',run_number_defining_re_MRI='Ser([0-9]*)_',run_number_defining_re_behavior='scan_([0-9]*)_',readme_text='',behavior_scan_pairing_info_path=None,scan_runs_to_skip_info_path=None,sort_behav_sessions_manually=True,skip_if_present=True):
	
	"""Use data present in the 'sourcedata' folder to populate the 'rawdata' folder. This function expects there to be an 'MRI' subfolder inside the 'sourcedata' folder, and will also
		see if there is a 'behavior' folder in 'rawdata'. User will get opportunity to interactively point to which folders in 'MRI' to use, and to identify
		the session order. Depending on whether behavior_scan_pairing_info_path argument is provided, either the file at that path determines which folder in 'behavior' corresponds to which folder in 'MRI',
		or the user can interactively indicate the correspondence. If any 'ses' folders are already present in a subject's rawdata folder, then it is
		assumed that the ones we're working on now are subsequent, so they will be added rather than overwriting existing 'ses' folders. The 'run' values in the functional files will match
		the ones in the original Dicom files; not the ones in the original behavioral files (but see notes for 'run_number_defining_re_behavior' argument). A 'dataset_description.json' file, with 
		contents specified by the dataset_description argument, will be placed at the root of the 'rawdata' subfolder unless one is already present.

		All data (MRI, and optionally behavioral) are expected to be organized such that there is one folder per session.
		MRI data are expected to be in Dicom format, and will be converted to Nifti in the process. It is fine if the 'MRI' folder in 'sourcedata' contains prep scans as well; those will be ignored
			automatically.
		Behavioral data are expected to be in BIDS compatible 'tsv' format already, but file names can be whatever as long as the run order corresponds to that of the MRI data. It's fine if
			the behavioral folder contains other files (not 'tsv') as well; those will be ignored.
		If you want '_events.json' files to be placed in rawdata, then you need to create one for each task and point to it using scan_types. See 'scan_types' below for details.
		Optionally creates a 'README' text file at the root of your 'rawdata' subfolder, with contents either provided by readme_text argument or simply copied over from 'README' file in 'sourcedata'
			subfolder. See 'readme_text' for details.
		Keeps track of which data in sourcedata end up being which data in rawdata, and writes that information to README at the top of rawdata folder.
	
		Note: this function is organized assuming that scans associated with a given task can be identified by the name the corresponding protocol had in the scanner console (i.e. the 'series_description'
			key in 'scan_types' argument). This assumption is not met in an experiment where one and the same protocol, named one and the same thing in the scanner console, may have been accompanied
			by various tasks for the participant.

	Arguments:
		base_of_BIDS_tree (string). Path of the base of the BIDS structure, directly in which 'sourcedata' folder is present.
		subjects (list of strings). Subject identifiers that will be used in the 'rawdata' folder (not including the 'sub-' part itself).
		scan_types (list of dictionaries). Identifies the types of scans that you want to include in the 'rawdata' folder. Possible dictionary keys: suffix, task, series_description, and 
			example_events_json_file, as explained in the following. Each dictionary needs to have at least a 'suffix' key and, unless that 'suffix' key that has the value 'T1w' or 'T2w',
			'task' key. If the 'suffix' key that has the value 'T1w' or 'T2w', then the scan type is interpreted as 'anat'; otherwise as 'func'. Optionally, the key 'series_description' can also be provided
			for individual dictionaries. This is a string that corresponds with the 'SeriesDescription' field in the json file prodiced by running dcm2niix on the scanner's Dicom files. This is what the
			scan type was called on the scanner console. If 'series_description' is not provided, then the user is given the option to select, for each element of scan_types, which of the available
			'SeriesDescription' values in the jsons applies. If 'example_events_json_file' is provided, then that should correspond to a file that is present in os.path.join(base_of_BIDS_tree,'sourcedata','behavioral').
			Copies of this file will then be placed as '_events.json' files in the 'func' subfolders.
		dataset_description (dict, optional). Will be used to create dataset_description.json at the root of the 'rawdata' subfolder, unless that file is already present.
		run_number_defining_re_MRI (string, optional). Regular expression that, from the name of a .nii file that dcm2niix produces out of the scanner's Dicoms, isolates the integer that indicates the number of
			this run, equivalent to the scan number seen in the terminal during the scanning session. Defaults to 'Ser([0-9]*)_'.
		run_number_defining_re_behavior (string, optional). Regular expression that, from the name of a .tsv file that contains a run's behavioral data, isolates the integer that indicates the number of
			this run. This number is usually not the same as the run number of the corresponding.nii file because the latter is influenced by numbers taken up by prep scans, anatomical scans, etc. But it is
			critical that the ordering of runs corresponds between MRI scans and behavioral files. Defailts to 'scan_([0-9]*)_'.
		readme_text (string, optional). If provided, a file called README will be created in the sourcedata folder, with this text in it. If not provided, then a README file will be copied from
			'sourcedata' subfolder if it exists there.
		behavior_scan_pairing_info_path (string, optional). If provided, then points to a .tsv file that indicates which behavioral subject name (left column) corresponds with which scanner folder(s)
			(all columns to the right of the leftmost one). Those scanner folders should be provided in the order (columns going right) in which you want the sessions to appear in the folder structure.
		scan_runs_to_skip_info_path (string, optional). If provided, then points to a .tsv file that indicates, for specific scan sessions, which MRI scan numbers should not be transferred from the 
			sourcedata folder to the rawdata folder. Leftmost column is the scan folder, all columns to the right of that are runs that should be skipped. This argument is useful if scans were re-run and/or terminated
			because of issues during scanning. This argument only takes care of removal of scan runs (not behavioral data), and pairing up between behavioral runs and scan runs is done later, after excluding the scans listed 
			in this tsv file.
		sort_behav_sessions_manually (boolean, optional). If set to True, then the user will get a prompt for each scan folder of a given participant, to allow them to select the corresponding behavioral
			folder manually. If set to False, then it is critical that the behavioral folders are named such, that alphabetical sorting will result in the correct session order per participant automatically.
		skip_if_present (boolean, optional): If set to True, then no new session folders will be added to a subject's raw folder if any session folders are present there already. If set to False, then all sessions present in 
			the sourcedata will be added as session folders in the rawdata folder, counting session numbers upward from the highest session number that is already present in the rawdata folder.
	
	Returns:
		Nothing.

	To do:
		-Will at some point probably need to be expanded to also work with physiology data and/or eye data. Perhaps allow an alternative way of defining the 'events.json' files: currently those can only be
		defined by providing the name of an example file that has manually (or otherwise) been created and placed in the 'sourcedata/behavior' folder; it would make some sense to allow the user to define it
		in the code instead.
		-I wouldn't quite trust this function if at some point you need to apply it to a new session of a given participant after having already applied it to earlier sessions of the same participant before. Not
		100 percent sure that numbering would all be fine then so be careful.
	"""
	
	BIDS_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	os.makedirs(os.path.join(base_of_BIDS_tree,'rawdata'), exist_ok=True)
	
	if dataset_description and not('dataset_description.json' in glob.glob1(os.path.join(base_of_BIDS_tree,'rawdata'),'*') ):
		
		with open(os.path.join(base_of_BIDS_tree,'rawdata','dataset_description.json'), "w") as write_file:
			json.dump(dataset_description, write_file, indent="")
	
	if behavior_scan_pairing_info_path:
		with open(behavior_scan_pairing_info_path) as f:
			behavior_scan_pairing_info=f.read()
		behavior_scan_pairing_info=[info_one_row.split(',') for info_one_row in behavior_scan_pairing_info.split('\n')]
		
	if scan_runs_to_skip_info_path:
		with open(scan_runs_to_skip_info_path) as f:
			scan_runs_to_skip_info=f.read()
		scan_runs_to_skip_info=[info_one_row.split(',') for info_one_row in scan_runs_to_skip_info.split('\n')]

	for subject in subjects:
		
		#Create relevant rawdata folder, loop over subjects as well as sessions for each subject:
		
		os.makedirs(os.path.join(base_of_BIDS_tree,'rawdata','sub-'+subject), exist_ok=True)
		
		if not('MRI' in glob.glob1(os.path.join(base_of_BIDS_tree,'sourcedata'),'*')):
			raise ValueError('We need a path called \'MRI\' in the \'sourcedata\' folder.')
		else:
			existing_raw_session_folders_this_sub=glob.glob1(os.path.join(base_of_BIDS_tree,'rawdata','sub-'+subject),'ses-*')
			
			if len(existing_raw_session_folders_this_sub)>0 and skip_if_present:
				BIDS_tools_logger.info('Not adding anything to rawdata folder of '+subject+' because at least one session is already present and skip_if_present argument is set to True')
			else:
				BIDS_tools_logger.info('Adding new session folders to rawdata folder of '+subject+', starting at session number '+str(len(existing_raw_session_folders_this_sub)+1))
				
				possible_scan_session_paths=[candidate for candidate in glob.glob(os.path.join(base_of_BIDS_tree,'sourcedata','MRI','*')) if os.path.isdir(candidate)]
			
				success=False
				if 'behavior_scan_pairing_info' in locals():		#if the info is available, then establish the pairing between behavioral subject ID and scan session folders based on that info...
				
					for info_one_row in behavior_scan_pairing_info:
						if info_one_row[0]==subject:
							source_MRI_paths=[os.path.join(base_of_BIDS_tree,'sourcedata','MRI',one_relative_path) for one_relative_path in info_one_row[1:] if len(one_relative_path)>0]
							success=True
							break
			
				if not(success):									#... otherwise let user do it manually
				
					prompt_string='\nWhich \'sourcedata\' folders correspond to the MRI sessions you want to select for subject '+subject+'? (The order of indices here will determine the order of the \'ses\' folders).'
					source_MRI_paths=general_tools.interactively_select_strings(possible_scan_session_paths, \
						prompt_string = prompt_string)
				
				for path_index,source_MRI_path in enumerate(source_MRI_paths):	#each source_MRI_path corresponds to one session
					
					runs_to_exclude=[]
					if 'scan_runs_to_skip_info' in locals():	#prepare to ignore certain run numbers (i.e. not transfer them from sourcedata to rawdata), if provided info indicates we should
						
						for info_one_scanner_folder in scan_runs_to_skip_info:
							if info_one_scanner_folder[0]==os.path.split(source_MRI_path)[1]:
								runs_to_exclude=info_one_scanner_folder[1:]
								break
								
					#Determine the contents of README file to be written to rawdata. But don't write anything yet because we'll
					#append information about the correspondence between data in sourcedata and rawdata.
					if 'README' in glob.glob1(os.path.join(base_of_BIDS_tree,'rawdata'),'*'):	#take the file already in rawdata if it's there.
				
						readme_file = open(os.path.join(base_of_BIDS_tree,'rawdata','README'),"r")
						readme_text=readme_file.read()
						readme_file.close()
	
					elif not(readme_text) and 'README' in glob.glob1(os.path.join(base_of_BIDS_tree,'sourcedata'),'*'):	#stick with readme_text argument unless the argument is empty and a file is present in sourcedata
		
						readme_file = open(os.path.join(base_of_BIDS_tree,'sourcedata','README'),"r")
						readme_text=readme_file.read()
						readme_file.close()
		
					readme_text+='\n'
			
					#Move and organize MRI data:
			
					session_number=path_index+len(existing_raw_session_folders_this_sub)+1
			
					target_path=os.path.join(base_of_BIDS_tree,'rawdata','sub-'+subject,'ses-'+str(session_number))
			
					os.makedirs(os.path.join(target_path,'func'))
					os.makedirs(os.path.join(target_path,'anat'))
			
					os.makedirs(os.path.join(source_MRI_path,'temp'), exist_ok=True)
		
					possible_Dicom_folders=[candidate for candidate in glob.glob(os.path.join(source_MRI_path,'*')) if os.path.isdir(candidate)]
			
					for possible_Dicom_folder in possible_Dicom_folders:	#Convert dicom folders to .nii files that are stored in a temporary place:
				
						possible_Dicom_files=glob.glob1(possible_Dicom_folder,'*.MRDC.*')
				
						if possible_Dicom_files:		#if this folder has no Dicom files then simply skip
							possible_Dicom_files.sort()
							general_tools.execute_shell_command(['dcm2niix','-o',os.path.join(source_MRI_path,'temp'), os.path.join(possible_Dicom_folder,possible_Dicom_files[0])],collect_response=False,wait_to_finish=True)
							readme_text+='\nConverted to nifti: '+os.path.join(possible_Dicom_folder,possible_Dicom_files[0])
					
					json_files=glob.glob(os.path.join(source_MRI_path,'temp','*.json'))		#dcm2niix automatically creates a .json when called
					file_name_roots=[os.path.split(os.path.splitext(this_json_file)[0])[1] for this_json_file in json_files]
					series_descriptions=[json.load(open(this_json_file))["SeriesDescription"] for this_json_file in json_files]
					unique_series_descriptions=list(set(series_descriptions))
					scan_types_deepcopy=copy.deepcopy(scan_types)		#We need to apply to following changes to a deepcopy because the same changes may not apply to each source_MRI_path
					for scan_type in scan_types_deepcopy:				#Make sure each scan_type has a series description: we need to know what
						#the type of scan we're talking about was called in the scanner console. Equivalently, what it's called in the .json produced when converting from .dicom to .nii.
						#Also, when running fill_rawdata_folder() on a number of different sessions then it's possible that scan_types contains some entries corresponding to scan types
						#not featured in a particular session. So here the user has the option of indicating that, as well.
						if not 'series_description' in scan_type:
							series_description_candidate=general_tools.interactively_select_strings(unique_series_descriptions+['Scan type is not present in this session.'],'For the MRI data in '+source_MRI_path+', which Series Description belongs to the scan type with the following properties: '+', '.join([key+': '+scan_type[key] for key in scan_type])+'?')
							while not(len(series_description_candidate)==1):
								series_description_candidate=general_tools.interactively_select_strings(unique_series_descriptions+['Scan type is not present in this session.'],'Don\'t listen to the other guy. You need to select exactly one index for this.')
					
							if not(series_description_candidate[0]==len(series_description_candidate)):	#else: scan_type['series_description'] remains undefined
								scan_type['series_description']=series_description_candidate[0]
				
					#we'll assemble all we need to know about each scan to place it correctly in the BIDS structure:	
					anat_func_values=[]			#does it go into the func folder or the anat folder?
					target_filenames=[]			#what filename to use in the rawdata folder? This also includes a task name and a run number if appropriate.
					target_json_filenames=[]	#and what filename for the accompanying json file?
					source_file_name_roots=[]	#and what is the file name like in the sourcedata folder?
					nifti_file_extensions=[]	#are these .niis or .nii.gzs?
			
					for scan_type in scan_types_deepcopy:
				
						if 'series_description' in scan_type:
							these_file_name_roots=[file_name_roots[index] for index, series_description in enumerate(series_descriptions) if series_description==scan_type['series_description']]
						else:
							these_file_name_roots=[]
					
						if these_file_name_roots:	#if there are any scans of this scan_type in this session
					
							run_number_strings=[re.search(run_number_defining_re_MRI,this_root)[1] for this_root in these_file_name_roots]
							
							these_file_name_roots=[these_file_name_roots[index] for index,value in enumerate(run_number_strings) if not value in runs_to_exclude]
							run_number_strings=[value for value in run_number_strings if not value in runs_to_exclude]
				
							os.path.splitext(glob.glob(os.path.join(source_MRI_path,'temp',these_file_name_roots[0]+'.nii*'))[0])[1]
					
							if os.path.splitext(glob.glob(os.path.join(source_MRI_path,'temp',these_file_name_roots[0]+'.nii*'))[0])[1]=='.nii':
								target_file_extension='.nii'
							else:
								target_file_extension='.nii.gz'
				
							if 'task' in scan_type and not(run_number_strings[file_index] in runs_to_exclude):	#functional scans
					
								entities = {
									'sub': subject,
									'ses': str(session_number),
									'task': scan_type['task'],
									'suffix': scan_type['suffix']
								}
					
								for file_index,file_name_root in enumerate(these_file_name_roots):
						
									anat_func_values+=['func']
									entities['run']=run_number_strings[file_index]
						
									entities['extension']=target_file_extension
									target_filenames+=[my_build_bids_filename(entities)]
									entities['extension']='.json'
									target_json_filenames+=[my_build_bids_filename(entities)]
									nifti_file_extensions+=[target_file_extension]
					
								success=True
					
					
							elif (scan_type['suffix'] in ['T1w','T2w']):	#anatomical ones
					
								entities = {
									'sub': subject,
									'ses': str(session_number),
									'suffix': scan_type['suffix']
								}
					
					
								for file_index,file_name_root in enumerate(these_file_name_roots):
						
									anat_func_values+=['anat']
						
									if len(these_file_name_roots)>1:
										entities['desc']=run_number_strings[file_index]
							
									entities['extension']=target_file_extension
									target_filenames+=[my_build_bids_filename(entities)]
									entities['extension']='.json'
									target_json_filenames+=[my_build_bids_filename(entities)]
									nifti_file_extensions+=[target_file_extension]
					
								success=True
					
							else:
					
								success=False	#if there is no suffix that indicates an anatomical scan, nor a task which is required for a functional, then this is no success.
						
							if success:
								source_file_name_roots+=these_file_name_roots
				
					#Now that we know all we need to know for each MRI file, let's do the actual moving and renaming.
					zipped_root_and_target_names = zip(source_file_name_roots,anat_func_values,target_filenames,target_json_filenames,nifti_file_extensions)
					# original line is commented out below this. the numpy.array way to doing this seemed to cause issues. this new way, zipping the lists, achieves the same thing. the issues I was getting may have been related to a different error, but regardless, this works now
					# for [source_file_name_root,anat_func_value,target_filename,target_json_filename,nifti_file_extension] in numpy.array([source_file_name_roots,anat_func_values,target_filenames,target_json_filenames,nifti_file_extensions]).T.tolist():
					for [source_file_name_root,anat_func_value,target_filename,target_json_filename,nifti_file_extension] in zipped_root_and_target_names:
				
						json_path_from=os.path.join(source_MRI_path,'temp',source_file_name_root+".json")
						json_path_to=os.path.join(target_path,anat_func_value,target_json_filename)
				
						task_if_any=re.search('.*task-(.*?)_.*',target_json_filename)
				
						if task_if_any:				#add 'Task' field to _bold.json file if there is a task.
							task=task_if_any[1]
					
							with open(json_path_from, "r") as read_file:
								json_data = json.load(read_file)
					
							json_data['TaskName']=task
					
							with open(json_path_to, "w") as write_file:
								json.dump(json_data, write_file, indent="")
						
						else:
					
							shutil.copy(json_path_from, json_path_to)
					
						mri_path_from=os.path.join(source_MRI_path,'temp',source_file_name_root+nifti_file_extension)
						mri_path_to=os.path.join(target_path,anat_func_value,target_filename)
				
						shutil.copy(mri_path_from, mri_path_to)
						readme_text+='\nMRI data copied from source to raw: '+source_file_name_root+nifti_file_extension+' to '+os.path.join(target_path,anat_func_value,target_filename)
	
					#Identify the corresponding behavioral data subfolder in the sourcedata folder, and organize those behavioral data in the rawdata folder as well:
					if 'behavior' in glob.glob1(os.path.join(base_of_BIDS_tree,'sourcedata'),'*'):
					
						candidate_behavioral_paths=[candidate for candidate in glob.glob(os.path.join(base_of_BIDS_tree,'sourcedata','behavior','*')) if os.path.isdir(candidate)]
					
						if sort_behav_sessions_manually:
					
							prompt_string='\nWhich behavioral \'sourcedata\' folder matches MRI \'sourcedata\' folder '+os.path.split(source_MRI_path)[1]+' and \'rawdata\' folder sub-'+subject+'/ses-'+str(session_number)+'?'
							source_behavioral_path=general_tools.interactively_select_strings(candidate_behavioral_paths, \
								prompt_string = prompt_string)[0]
					
						else:
						
							source_behavioral_paths_this_subject=[one_path for one_path in candidate_behavioral_paths if subject in os.path.split(one_path)[1]]
							source_behavioral_paths_this_subject.sort()
							source_behavioral_path=source_behavioral_paths_this_subject[path_index]		#this is the line that makes it so that you need to make sure the behavioral session folders for a given participant are named such, that alphabetical sorting corresponds with the sorting of the functional folders, unless sort_behav_sessions_manually is set to True
				
						source_behavioral_filenames=glob.glob1(source_behavioral_path,'*.tsv')
						source_behavioral_filenames.sort(key=lambda one_filename:int(re.search(run_number_defining_re_behavior,one_filename)[1]))	#sort by run number
				
						target_behavioral_filenames=[re.sub('_bold.json','_events.tsv',this_json_name) for this_json_name in target_json_filenames if '_task' in this_json_name]
				
						target_behavioral_filenames.sort(key=lambda one_filename:int(re.search('.*_run-([0-9]*)',one_filename)[1]))			#sort by run number
					
						if not(len(source_behavioral_filenames)==len(target_behavioral_filenames)):
							raise Exception("There is a mismatch for subject "+subject+" between the number of behavioral files and the number of (what should be) corresponding scanner files.")	
						
						zipped_root_and_target_names_behavioral = zip(source_behavioral_filenames,target_behavioral_filenames)
						# the original line is commented out just below this. the numpy.array way to doing this seemed to cause issues. this new way, zipping the lists, achieves the same thing. the issues I was getting may have been related to a different error, but regardless, this works now
						# for [source_behavioral_filename,target_behavioral_filename] in numpy.array([source_behavioral_filenames,target_behavioral_filenames]).T.tolist():
						for [source_behavioral_filename,target_behavioral_filename] in zipped_root_and_target_names_behavioral:
							shutil.copy(os.path.join(source_behavioral_path,source_behavioral_filename), os.path.join(target_path,'func',target_behavioral_filename))
							readme_text+='\nBehavioral data copied from source to raw: '+os.path.join(source_behavioral_path,source_behavioral_filename)+' to '+os.path.join(target_path,'func',target_behavioral_filename)
						
							#If an example _events.json file corresponding to this 'task' is defined in scan_types, then place correctly named copies of that file in the correct rawdata subfolder:
							this_task=re.search('.*_task-(.*?)_.*',target_behavioral_filename)[1]
					
							example_events_json_file=''
							for scan_type in scan_types_deepcopy:
								try:
									if scan_type['task']==this_task:
										example_events_json_file=scan_type['example_events_json_file']
										break
								except KeyError:
									pass
					
							if example_events_json_file:
								source_events_json_fullpath=os.path.join(base_of_BIDS_tree,'sourcedata','behavior',example_events_json_file)	
								target_events_json_fullpath=os.path.join(target_path,'func',os.path.splitext(target_behavioral_filename)[0]+'.json')	
								shutil.copy(source_events_json_fullpath, target_events_json_fullpath)
						
					readme_text+='\n'
					shutil.rmtree(os.path.join(source_MRI_path,'temp'))	#remove the 'nii' files from /sourcedata. The Dicom files will stay there.

					readme_file = open(os.path.join(base_of_BIDS_tree,'rawdata','README'),"w")
					readme_file.write(readme_text)
					readme_file.close()

def fill_sourcedata_folder(base_of_BIDS_tree,holding_dir_info_mri=[],raw_data_identifying_string_mri='',holding_dir_info_behavior=[],raw_data_identifying_string_behavior='',holding_dir_info_code=[],raw_data_identifying_string_code='',readme_text=''):

	"""Organize data in a 'sourcedata' folder at the base of the BIDS tree. You can use this to create the initial 'sourcedata' folder, 
		to copy data from holding areas to a 'sourcedata' folder, or to do both. User will get an opportunity to manually make a subselection
		of candidate MRI filed and behavioral files to be copied (because there can be impractically many). But not of code folders, which are explicitly provided as arguments (if that option is used).
		The organization created/assumed by this function is of a 'sourcedata' folder within base_of_BIDS_tree, and within 'sourcedata' three further subfolders
		called 'MRI','behavior', and, optionally, 'code'.
	
	Arguments:
		base_of_BIDS_tree (string). Path of the base of the BIDS structure, directly in which the 'sourcedata' folder will be placed.
		holding_dir_info_mri (list of strings, optional). If length is 1, then [path of local holding directory]. If length is 3, then [server path of holding
			directory, server username, server address].
		raw_data_identifying_string_mri (string, optional). Search pattern, including optional wildcards using linux syntax, to narrow down the listing of to-be-copied file names 
			or folder names in the holding area.
		holding_dir_info_behavior (list of strings, optional). Same as holding_dir_info_mri, but for behavioral data files/folders.
		raw_data_identifying_string_behavior (string, optional). Same as raw_data_identifying_string_mri, but for behavioral data files/folders.
		holding_dir_info_code (list of strings, optional). Same as holding_dir_info_mri, but for code files/folders.
		raw_data_identifying_string_code (string, optional). Same as raw_data_identifying_string_mri, but for code files/folders.
		readme_text (string, optional). If provided, a file called README will be created in the sourcedata folder, with this text in it. Will not overwrite existing file.
		
	Returns:
		nothing
	
	To do:
		Currently, base_of_BIDS_tree needs to be a local folder. if it is a server folder, the code probably needs to be altered.
	"""
	
	BIDS_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
		
	os.makedirs(os.path.join(base_of_BIDS_tree,'sourcedata'), exist_ok=True)
	
	#-----------------------						
	#Create a README file within the sourcedata folder. Do not overwrite existing.
	if readme_text and not( 'README' in glob.glob1(os.path.join(base_of_BIDS_tree,'sourcedata'),'*') ):
		readme_file = open(os.path.join(base_of_BIDS_tree,'sourcedata','README'),"w")
		readme_file.write(readme_text)
		readme_file.close()

	#----------------------
	#Move all data associated with a scan or scans to a 'sourcedata' folder within base_of_BIDS_tree. This includes scanner data files from the server as well as behavioral data files and 
	#folder(s) with code as used during scanning, both to be found locally, usually.
	
	for data_kind_index in range(3):
		sourcedata_subdir_name=['MRI','behavior','code'][data_kind_index]
		this_holding_dir_info=[holding_dir_info_mri,holding_dir_info_behavior,holding_dir_info_code][data_kind_index]
		this_raw_data_indentifying_string=[raw_data_identifying_string_mri,raw_data_identifying_string_behavior,raw_data_identifying_string_code][data_kind_index]
	
		local_target_dir=os.path.join(base_of_BIDS_tree,'sourcedata',sourcedata_subdir_name)
		os.makedirs(local_target_dir, exist_ok=True)
		
		if this_holding_dir_info:
			
			if len(this_holding_dir_info)==3:
				[server_holding_dir,server_username,server_address]=this_holding_dir_info
				general_tools.copy_files_with_interactive_selection([server_holding_dir,server_username,server_address],[[os.path.join(base_of_BIDS_tree,'sourcedata',sourcedata_subdir_name)]],raw_data_identifying_string=this_raw_data_indentifying_string)
			elif len(this_holding_dir_info)==1:
				local_holding_dir=this_holding_dir_info[0]
				general_tools.copy_files_with_interactive_selection([local_holding_dir],[[os.path.join(base_of_BIDS_tree,'sourcedata',sourcedata_subdir_name)]],raw_data_identifying_string=this_raw_data_indentifying_string)

def get_anatomical_path(layout,this_sub,space='T1w'):
	"""Search the fMRIPrep 'derivatives' folder of a BIDS tree to identify the path of a T1-weighted anatomical of a given subject, and return it.
		
	Arguments:
		layout (bids.BIDSLayout object): created using the raw data folder as its first argument and the fMRIPrep derivatives folder as the named 'derivatives' argument.
		this_sub (string): subject identifier, without the 'sub-' part.
		
	Returns:
		anatomical_path (string): absolute path to an anatomical file.
	"""
	
	if space=='T1w':
		return [candidate_filename for candidate_filename in layout.get(scope='derivatives', return_type='file', suffix='T1w', extension=['nii', 'nii.gz']) if re.search("sub-" + this_sub, candidate_filename) and not(re.search("_space-", candidate_filename)) ][0]
	else:
		return [candidate_filename for candidate_filename in layout.get(scope='derivatives', return_type='file', space=space, suffix='T1w', extension=['nii', 'nii.gz']) if re.search("sub-" + this_sub, candidate_filename)][0]

def my_build_bids_filename(entities):
	"""Create a BIDS-style filename or foldername based on entities provided. Similar in purpose to pybids' build_path method, as well as mne_bids' BIDSPath method, but more basic.
		
	Arguments:
		entities (dictionary): each key is an entity name, preferably one allowed in BIDS (https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html), and each value is its value. Except for
			the optional special key 'suffix' and the optional special key 'extension'. 'suffix' refers to the part of the filename right after the final '_' but 
			before the extension. 'extension' refers to the filename part starting with the leftmost '.' (https://bids-specification.readthedocs.io/en/stable/02-common-principles.html).
			'sub' is a required key in entities unless no extension is provided (indicating this is a derivatives folder name and not a subject specific filename), and so is 'task' unless 'suffix' or 'space' is 'T1w' or 'T2w'. If 'extension' key is not provided then '.nii.gz' is used as long as 'sub' is also present (we only want extensions on filenames not derivatives folders).
	
	Returns:
		filename (string): filename or derivatives foldername that has been constructed based on the information in entities, and that tries to match BIDS expectations.
	
	"""

	BIDS_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function

	filename_beginning = ''
	filename_ending = ''
		
	try:
		filename_beginning='sub-'+entities['sub'] + '_'					#subject
	except KeyError:

		all_ok = False	

		if ('extension' not in entities):
			all_ok=True

		if all_ok:
			pass
		else:
			raise Exception("Entities dictionary must contain 'sub', unless there is no extension (indicates a derivative folder name rather than a subject specific file)")	
		
	if 'ses' in entities:
		filename_beginning+='ses-'+entities['ses'] + '_'			#session
	
	try:
		filename_beginning+='task-'+entities['task'] + '_'			#task
	except KeyError:
		
		all_ok=False
		
		if ('suffix' in entities):
			if entities['suffix'] in ['T1w','T2w']:
				all_ok=True
				
		if ('space' in entities):									
			if entities['space'] in ['T1w','T2w']:
				all_ok=True
		
		if all_ok:
			pass
		else:
			raise Exception("Entities dictionary must contain 'task', unless either suffix or space is 'T1w or 'T2w'")
	
	if 'space' in entities:
		filename_beginning+='space-'+entities['space'] + '_'			#space
	if 'fROI' in entities:
		filename_beginning+='fROI-'+entities['fROI'] + '_'				#functional ROI
	if 'aROI' in entities:
		filename_beginning+='aROI-'+entities['aROI'] + '_'				#anatomical ROI
	if 'deco' in entities:
		filename_beginning+='deco-'+entities['deco'] + '_'				#classifier type used (for instance: linear SVM)
	
	if 'desc' in entities:
		filename_ending+='desc-'+entities['desc']+ '_'
	if 'var' in entities:
		filename_ending+='var-'+entities['var']+ '_'
	
	for this_key in entities:
		if not this_key in ['sub','ses','task','space','fROI','aROI','deco','desc','var','suffix','extension']:
			filename_beginning+=this_key+'-'+entities[this_key] + '_'
	
	if 'suffix' in entities:
		filename_ending+=entities['suffix']
	else:
		filename_ending=filename_ending[:-1]

	if 'extension' in entities:
		filename_ending+=entities['extension']
	elif ('extension' not in entities) and ('sub' in entities):
		filename_ending+='.nii.gz'

	filename=filename_beginning+filename_ending
		
	return filename	

def my_get_bids_entities(the_path):
	"""Get BIDS entities based on file name or path provided. Similar in purpose to .get_entities() method of BIDSFile objects, but I had issues with
		that method.
		
	Arguments:
		the_path (string): path or filename. Can be absolute path, relative path, or just filename.
	
	Returns:
		entities (dictionary): entity names and their values.
	
	"""

	BIDS_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	file_name_no_extension=(os.path.split(the_path)[1]).split('.')[0]
	
	entities={}
	for component in file_name_no_extension.split('_'):
		if '-' in component:
			[entity,value]=component.split('-')
		else:
			[entity,value]=['suffix',component]
		entities[entity]=value
	
	return entities
	
def prepare_derivatives_folder(base_of_BIDS_tree, main_script_folder_name,readme_text='', variant_details = ''):
	"""Create a derivatives subfolder (if it doesn't yet exist) with a name and location that parallel those of the script calling this function. Inside the derivatives folder create a README file so that you can explain what's in that folder.
		On each call to this function, regardless of whether the folder already existed, also append to the README file to specify which scripts wrote to the derivatives folder and when. This function should be called by any scripts that write
		to this derivatives subfolder.
	
	Arguments:
		base_of_BIDS_tree (string). Path of the base of the BIDS structure, directly in which the 'derivatives' can be found.
		main_script_folder_name (string): name (not full path) of the folder within which all scripts of this project are contained. The placement of output_path inside the /derivatives folder is set such,
			that it parallels the placement inside main_script_folder_name of the script that calls this function.
		readme_text (string): text that will be written to README file the first time README is created. Should specify what this derivatives subfolder is about. After this first creation, readme_text is not written again, 
			but rather mention will be appended of which scripts wrote to this subfolder and when.
		variant_details (string, optional): either a path to a folder to derive variant information from or a phrase to be used as the variant information. For example, a generic decoding script (one that does not have a desc) might set 
			variant_details to be the path to the derivatives folder containing the t-stat values to be used in the decoding. In that case, the desc from the t-stat path will be used as a var keyword to distinguish between variations of outputs all
			generated by the same script. If the decoding script is not generic and contains a desc, a path or keyword may still be passed and will be assigned to the var keyword. This usage is not limited to decoding scripts. 
			To add variant details to a derivatives folder name, simply pass what you want the variant details to be. For example, pass variant_details = "customHRF" to include "var-customHRF" in the derivatives folder name. 
			If variant_details is not passed, the resulting derivates folder name will be identitical to the calling script (minus the .py)
	
	Returns:
		derivatives_path: absolute path of the derivatives (sub)folder that the calling script should be writing to.
	
	"""
	
	BIDS_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	calling_script_full_path=re.search(".*file \'(.*)\'",str(inspect.stack()[1][0]))[1]
	
	"""
	apparently, breaking a full path into its individidual components (drive, folders, filename) in a reliable way is a bit
	cumbersome. That's what's being done in the following while loop and associated lines. Adapted from:
	https://stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python
	"""
	path=calling_script_full_path
	calling_script_folder_tree = []
	while True:
		path_except_last_element, last_element = os.path.split(path)
		if last_element != "":
			calling_script_folder_tree.append(last_element)
			path=path_except_last_element
		else:
			break
	calling_script_folder_tree.reverse()
	
	main_script_folder_index_in_tree=[index for index,value in enumerate(calling_script_folder_tree) if value==main_script_folder_name][0]
	
	"""add subfolders inside /derivatives based on any subfolders that are deeper than main_script_folder_name inside calling_script_folder_tree"""
	derivatives_path_under_construction=os.path.join(base_of_BIDS_tree,'derivatives')
	for subfolder in calling_script_folder_tree[main_script_folder_index_in_tree+1:len(calling_script_folder_tree)-1]:	
		derivatives_path_under_construction=os.path.join(derivatives_path_under_construction,subfolder)
	
	"""if not creating a variant, name derivatives folder the same as the calling file"""
	final_subfolder_name_under_construction=calling_script_folder_tree[-1].split('.')[0]
	
	"""if we are creating a variant, add 'var' to entities with given variant_name value"""
	if variant_details != '':
		#if final_subfolder_name_under_construction already contains a 'var' entity before even adding my_var (because the name of the calling file did), 
		#then append my_var to the value of that 'var'.
		existing_entities=my_get_bids_entities(final_subfolder_name_under_construction)
	
		my_new_entities = my_get_bids_entities(final_subfolder_name_under_construction)
		if '/' in variant_details:
			# variant_details string is a path
			my_variant_entities = my_get_bids_entities(variant_details.split('/').pop())
			if 'var' in my_variant_entities:
				my_var = my_variant_entities['var']
			elif 'desc' in my_variant_entities:
				my_var = my_variant_entities['desc']
			else:
				my_var = ''
		else:
			# variant_details string is a phrase
			my_var = variant_details

		if 'var' in existing_entities and not(my_var == ''):
			if 'desc-' in variant_details:
				my_new_entities['desc'] = variant_details[5:]
				my_var = ''
			else:
				my_var=existing_entities['var']+my_var[0].upper()+my_var[1:]					

		if not my_var=='':
			my_new_entities['var'] = my_var
		
		final_subfolder_name_under_construction=my_build_bids_filename(my_new_entities)

	derivatives_path=os.path.join(derivatives_path_under_construction,final_subfolder_name_under_construction)
	os.makedirs(derivatives_path, exist_ok=True)
	
	"""write the overall readme_text, but only if the README file doesn't exist yet"""
	if not( 'README' in glob.glob1(derivatives_path,'*') ):
		readme_file = open(os.path.join(derivatives_path,'README'),"w")
		readme_file.write(readme_text)
		readme_file.close()

	"""each time this function is called by a script, add mention of that to the README file, even if the file already existed"""
	now = datetime.now()
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
	readme_file = open(os.path.join(derivatives_path,'README'),"a")
	readme_file.write('\n\nA script named '+calling_script_full_path+' wrote to this derivatives folder at the following time: '+date_time)
	readme_file.close()	
	
	"""create dataset_description.json"""
	if not('dataset_description.json' in glob.glob1(derivatives_path,'*') ):
	
		dataset_description={'Name':final_subfolder_name_under_construction,\
		'BIDSVersion': 'v1.6.0',
		'DatasetType': 'derivative',
		'GeneratedBy': [
			{
				"Name": calling_script_full_path,
				"Version": date_time,
				 "CodeURL": "https://github.com/janbrascamp/analysis_code_Templeton_2a"
			 }
		 ]
		}
		
		with open(os.path.join(derivatives_path,'dataset_description.json'), "w") as write_file:
			json.dump(dataset_description, write_file, indent="")
	
	return derivatives_path