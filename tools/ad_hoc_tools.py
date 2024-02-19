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
import scipy.io

from . import general_tools, fMRI_tools

ad_hoc_tools_logger = general_tools.get_logger(__name__)

def convert_behavioral_data_to_bids_tsv_for_face_project(behavioral_data_path,face_position_file_path_sz,face_position_file_path_hc,scan_folder_identifier_string='*'):
	"""This function takes behavioral data produced by the script TempIntegMRI_v12_dot_FR_MRI.m during scan sessions for the face integration project in the Thakkar lab (2022 or thereabouts), and
	puts the relevant data in TSV files that are BIDS compatible. Skips instances where the TSV file in questions is already present.
	
	Arguments:
		behavioral_data_path (string). Absolute path of the folder where subfolders for individual participants' behavioral data can be found. Typically, if we're aiming
			for a BIDS compliant data set, this is the absolute path of the sourcedata/behavior folder.
		face_position_file_path_sz (string): Absolute path of comma separated file that contains the faces' x, y, z positions in a 3D similarity space. In the file, each row corresponds to one face (ascending order
			of numbering in file name) and each column corresponds with one of the 3 dimensions (x, y, z).
		face_position_file_path_hc (string): Same as face_position_file_path_sz but for healthy controls.
		scan_folder_identifier_string (string,optional). String (optionally including wildcard '*') that identifies subfolders within each given participant's subfolder in behavioral_data_path,
			that correspond to individual scans.

	Returns:
		Nothing, but writes .tsv files to the same individual participants' subfolders within behavioral_data_path
	"""
	
	ad_hoc_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	all_participant_folders=[element for element in glob.glob1(behavioral_data_path,'*') if os.path.isdir(os.path.join(behavioral_data_path,element))]

	for participant_folder in all_participant_folders:
		
		scan_subfolders=[candidate for candidate in glob.glob1(os.path.join(behavioral_data_path,participant_folder),scan_folder_identifier_string) if os.path.isdir(os.path.join(behavioral_data_path,participant_folder,candidate))]
		for scan_subfolder in scan_subfolders:
			
			output_filename=scan_subfolder+'_behavioral_data_in_BIDS_format.tsv'
			
			if output_filename in glob.glob1(os.path.join(behavioral_data_path,participant_folder),'*'):
				ad_hoc_tools_logger.info('Skipping creation of '+output_filename+' because the file is already present at path '+os.path.join(behavioral_data_path,participant_folder))
			else:
			
				data_file_names=glob.glob1(os.path.join(behavioral_data_path,participant_folder,scan_subfolder),"trialstruct_acc_*.mat")
				if len(data_file_names)>1:
					raise Exception("I expected there to be only one viable trial_struct_acc_*.mat file inside "+os.path.join(behavioral_data_path,participant_folder,scan_subfolder)+" but that doesn't seem to be the case.")
				
				data_file_name=data_file_names[0]
					
				full_data_object=scipy.io.loadmat(os.path.join(behavioral_data_path,participant_folder,scan_subfolder,data_file_name))

				actual_data=full_data_object['trialstruct'][0]

				all_keys=list(actual_data.dtype.fields.keys())

				data_dictionary={}
			
				#now follows an awful piece of code to turn an awful object, created by reading in a Matlab .mat file, into an
				#organized dictionary
				for key_index, key in enumerate(all_keys):
					data_dictionary[key]=[]
					for data_element in actual_data:
						candidate_to_be_added=data_element[key_index][0]
						if isinstance(candidate_to_be_added,numpy.ndarray):
							if len(candidate_to_be_added)==1:
								if isinstance(candidate_to_be_added[0],numpy.void):
				
									all_keys_of_sub_dictionary=list(candidate_to_be_added.dtype.fields.keys())
									sub_dictionary={}
									for data_index, data_value in enumerate(candidate_to_be_added[0]):
					
										if isinstance(data_value,numpy.ndarray):
											if len(data_value[0])==1:
												sub_dictionary[all_keys_of_sub_dictionary[data_index]]=data_value[0][0]
											else:
												sub_dictionary[all_keys_of_sub_dictionary[data_index]]=data_value[0]
										else:
											sub_dictionary[all_keys_of_sub_dictionary[data_index]]=data_value
						
									data_dictionary[key]+=[sub_dictionary]	
			
								else:	
									data_dictionary[key]+=[candidate_to_be_added[0]]
							elif len(candidate_to_be_added)==0:
								data_dictionary[key]+=[candidate_to_be_added]
							else:
								if isinstance(candidate_to_be_added[0],numpy.void):
									all_keys_of_sub_dictionary=list(candidate_to_be_added.dtype.fields.keys())
									sub_dictionary={}
									for sub_key_index,sub_key in enumerate(all_keys_of_sub_dictionary):
										sub_dictionary[sub_key]=[]
										for one_element in candidate_to_be_added:
											if isinstance(one_element[sub_key_index][0],numpy.ndarray) and len(one_element[sub_key_index][0])==1:
												sub_dictionary[sub_key]+=[one_element[sub_key_index][0][0]]
											else:
												sub_dictionary[sub_key]+=[one_element[sub_key_index][0]]
									data_dictionary[key]+=[sub_dictionary]
								else:
									data_dictionary[key]+=[candidate_to_be_added]

						else:
							data_dictionary[key]+=[candidate_to_be_added]
			
				first_TR_time=data_dictionary['trs'][0][0]			#this is the very first time slice of the functional data, and the TR that triggers the experiment
				if len(data_dictionary['trs'][0])==4:
					first_used_TR_time=data_dictionary['trs'][0][-1]	#this is the TR that ends the waiting period at the start of the experiment, and some timestamps in the behavioral output file are defined relative to the moment of this TR
				else:
					 raise Exception("I expected the first 4 TRs to all be in the 0th element of data_dictionary['trs'] but that doesn't seem to be the case.")
			
				#CHECK WITH JESSICA: IS THE ORDER OF THE XYZ POSITIONS HERE IN ASCENDING ORDER OF FILENAME?!	
				with open(face_position_file_path_sz) as f:
					face_space_info_sz=f.read()
				face_coordinate_strings_sz=[string_with_coordinates_one_face.split(',') for string_with_coordinates_one_face in face_space_info_sz.split('\n') if len(string_with_coordinates_one_face)>0]
			
				with open(face_position_file_path_hc) as f:
					face_space_info_hc=f.read()			
				face_coordinate_strings_hc=[string_with_coordinates_one_face.split(',') for string_with_coordinates_one_face in face_space_info_hc.split('\n') if len(string_with_coordinates_one_face)>0]
			
				output_column_headers=['onset','duration','trial_type','stim_file','face_x_loc_in_similarity_space_HC','face_y_loc_in_similarity_space_HC','face_z_loc_in_similarity_space_HC','face_x_loc_in_similarity_space_SZ','face_y_loc_in_similarity_space_SZ','face_z_loc_in_similarity_space_SZ','dot_side','response_side']
			
				trial_data_array=[]
			
				for so_called_trial_index,one_face_stimulus_used in enumerate(data_dictionary['image']):	#'so-called' because during blank periods there's not actually any trial structure visible to the participant
					if one_face_stimulus_used:		#this will evaluate to false for blank trials (where one_face_stimulus_used is an empty array)
						this_onset=data_dictionary['stimtimes'][so_called_trial_index][0]-first_TR_time
						this_duration=data_dictionary['stimtimes'][so_called_trial_index][-1]-first_TR_time-this_onset+numpy.median([data_dictionary['stimtimes'][so_called_trial_index][frame_index]-data_dictionary['stimtimes'][so_called_trial_index][frame_index-1] for frame_index in range(1,len(data_dictionary['stimtimes'][so_called_trial_index]))])	#add one (median) frame duration because these are frame onset times
						this_trial_type='face_image'
					
						this_stim_file=one_face_stimulus_used.split('\\')[-1]
					
						this_image_index=int((this_stim_file.split('.')[0]).split('_')[-1])-1		#-1 because file name numbering starts at 1
					
						this_face_x_loc_in_similarity_space_hc=face_coordinate_strings_hc[this_image_index][0]
						this_face_y_loc_in_similarity_space_hc=face_coordinate_strings_hc[this_image_index][1]
						this_face_z_loc_in_similarity_space_hc=face_coordinate_strings_hc[this_image_index][2]
					
						this_face_x_loc_in_similarity_space_sz=face_coordinate_strings_sz[this_image_index][0]
						this_face_y_loc_in_similarity_space_sz=face_coordinate_strings_sz[this_image_index][1]
						this_face_z_loc_in_similarity_space_sz=face_coordinate_strings_sz[this_image_index][2]
					
						this_dot_side=data_dictionary['trial_type'][so_called_trial_index]
					
						this_response_side='n/a'
					
						trial_data_array+=[[str(this_onset),str(this_duration),this_trial_type,this_stim_file,this_face_x_loc_in_similarity_space_hc,this_face_y_loc_in_similarity_space_hc,this_face_z_loc_in_similarity_space_hc,this_face_x_loc_in_similarity_space_sz,this_face_y_loc_in_similarity_space_sz,this_face_z_loc_in_similarity_space_sz,this_dot_side,this_response_side]]
			
				for element in data_dictionary['trs']:
					if type(element)==numpy.float64:
						one_array_of_TRs=numpy.array([element])
					else:
						one_array_of_TRs=element
					
					for one_TR in one_array_of_TRs:
					
						this_onset=one_TR-first_TR_time
						this_duration='n/a'
						this_trial_type='TR'
						this_stim_file='n/a'
						this_face_x_loc_in_similarity_space_hc='n/a'
						this_face_y_loc_in_similarity_space_hc='n/a'
						this_face_z_loc_in_similarity_space_hc='n/a'
						this_face_x_loc_in_similarity_space_sz='n/a'
						this_face_y_loc_in_similarity_space_sz='n/a'
						this_face_z_loc_in_similarity_space_sz='n/a'
						this_dot_side='n/a'
						this_response_side='n/a'
				
						trial_data_array+=[[str(this_onset),this_duration,this_trial_type,this_stim_file,this_face_x_loc_in_similarity_space_hc,this_face_y_loc_in_similarity_space_hc,this_face_z_loc_in_similarity_space_hc,this_face_x_loc_in_similarity_space_sz,this_face_y_loc_in_similarity_space_sz,this_face_z_loc_in_similarity_space_sz,this_dot_side,this_response_side]]
					
				for responses_one_so_called_trial in data_dictionary['resps']:	#'so-called' because during blank periods there's not actually any trial structure visible to the participant
					if type(responses_one_so_called_trial)==dict:
					
						time_or_times_relative_to_first_used_TR=responses_one_so_called_trial['time_exp']
					
						if type(time_or_times_relative_to_first_used_TR) == numpy.float64:
							times_relative_to_first_used_TR=numpy.array([time_or_times_relative_to_first_used_TR])
							responses=numpy.array([responses_one_so_called_trial['response']])
						else:
							times_relative_to_first_used_TR=time_or_times_relative_to_first_used_TR
							responses=responses_one_so_called_trial['response']
						
							if responses=='Otherkey':
								times_relative_to_first_used_TR=numpy.array([time_or_times_relative_to_first_used_TR])
								responses=numpy.array([responses])
					
						for key_in_trial_index in range(len(times_relative_to_first_used_TR)):
						
							if not (responses[key_in_trial_index]=='Otherkey'):
								this_onset=times_relative_to_first_used_TR[key_in_trial_index]+first_used_TR_time-first_TR_time
								this_duration=.5		#arbitrarily set duration of key press to 0.5 s
								this_trial_type='button'
								this_stim_file='n/a'
								this_face_x_loc_in_similarity_space_hc='n/a'
								this_face_y_loc_in_similarity_space_hc='n/a'
								this_face_z_loc_in_similarity_space_hc='n/a'
								this_face_x_loc_in_similarity_space_sz='n/a'
								this_face_y_loc_in_similarity_space_sz='n/a'
								this_face_z_loc_in_similarity_space_sz='n/a'
								this_dot_side='n/a'
								this_response_side=responses[key_in_trial_index]

								trial_data_array+=[[str(this_onset),str(this_duration),this_trial_type,this_stim_file,this_face_x_loc_in_similarity_space_hc,this_face_y_loc_in_similarity_space_hc,this_face_z_loc_in_similarity_space_hc,this_face_x_loc_in_similarity_space_sz,this_face_y_loc_in_similarity_space_sz,this_face_z_loc_in_similarity_space_sz,this_dot_side,this_response_side]]

				f = open(os.path.join(behavioral_data_path,participant_folder,output_filename), "w")
				f.write('\t'.join(output_column_headers) + '\n')
				for one_event_info in trial_data_array:
					f.write('\t'.join([str(element) for element in one_event_info]) + '\n')
				f.close()
	