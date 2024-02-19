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

def get_logger(name):
	"""Create a logger instance using logging.getLogger. Code adapted from https://stackoverflow.com/questions/45701478/log-from-multiple-python-files-into-single-log-file-in-python
	
	Arguments:
		name (string): used to specify which file get_logger was called from, which is then visible in the log file.
	
	Returns: 
		logger (logger instance): the logger instance
	"""
	
	name_of_importing_file=os.path.splitext(os.path.split(inspect.stack()[-1].filename)[-1])[0]
	
	log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'
	logging.basicConfig(level=logging.DEBUG,
						format=log_format,
						filename=name_of_importing_file+'_last_run.log',
						filemode='w')
	
	console = logging.StreamHandler()
	console.setLevel(logging.DEBUG)
	console.setFormatter(logging.Formatter(log_format))
	logging.getLogger(name).addHandler(console)
	logger=logging.getLogger(name)
	#logger.propagate = False
	return logger

general_tools_logger = get_logger(__name__)

#------------------------------------------

def write_dictlist_to_tsv(data,output_file_path,tsv_headers=[],write_mode='a',separator='\t'):
	"""Take data from a list of dictionaries and write those data to a .tsv file or .csv file, with data from each dict creating one row, and with
		column headers at the top.
	
	Arguments:
		data (list of dicts). Each entry will form a row.
		output_file_path (string). Absolute path to output file.
		tsv_headers (list of lists of strings, optional): Defines the column headers in the tsv/csv file, from left to right. Each of the headers also needs to correspond to a key in each of the
			dicts in data. If tsv_headers is not provided, then the keys of the first dict in data are used.
		write_mode (string, optional): 'mode' argument that is passed to python's write function. The write function supports 'w', 'x' and 'a'.
		separator (string, optional): separator between entries in the tsv/csv file.
		
	Returns:
		nothing
	"""
	
	general_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function

	if not(tsv_headers):
		tsv_headers=[element for element in data[0].keys()]
		
	with open(output_file_path, write_mode) as f:
		f.write(separator.join(tsv_headers)+'\n')
		for data_one_row in data:
			f.write(separator.join([str(data_one_row[key]) for key in tsv_headers])+'\n')
		
			

def copy_files_with_interactive_selection(source_dir_info,target_dirs_info,raw_data_identifying_string=''):
	"""Copy data from one place to one or more other places. This function will provide a full, numbered, file listing of the source folder, optionally using a string with wildcards 
		to narrow down the listing. The user can choose which data to copy. One or more target folders can be defined. Files will be copied, not moved,
		so they also remain at the original location. Will not overwrite files/folders that are already present.
	
	Arguments:
		source_dir_info (list of strings). Identifies location of data in the source area. If length is 3, then [path of on-server source directory, server username, server address].
			If length is 1, then [path of local source directory].
		target_dirs_info (list of lists of strings): Identifies one or more locations for data to be copied to. For each element: if length is 3, then [path of on-server target directory,
			server username, server address]. If length is 1, then [path of local target directory].
		raw_data_identifying_string (string,optional): Search pattern, including optional wildcards using linux syntax, to narrow down the listing of candidate raw filenames in the
			holding area.
		
	Returns:
		nothing
	
	To do:
		Currently, the only options are to move data from a server to elsewhere on the same server or to a local folder, and
		to move data from a local folder to another local folder. Moving between servers and uploading to a server not currently supported.
	"""

	general_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	source_path=source_dir_info[0]
	
	if len(source_dir_info)==3:
		source_server_user_name=source_dir_info[1]
		source_server_address=source_dir_info[2]
		source_file_paths_available=execute_shell_command(['ls',os.path.join(source_path,raw_data_identifying_string)],server_username=source_server_user_name,server_address=source_server_address)
	elif len(source_dir_info)==1:
		source_server_user_name=''
		source_server_address=''
		
		if not raw_data_identifying_string:
			wildcards='*'
		else:
			wildcards=raw_data_identifying_string
		
		source_file_paths_available=glob.glob(os.path.join(source_path,wildcards))
		
	if source_file_paths_available:
	
		selected_source_file_paths=interactively_select_strings(source_file_paths_available)
		
		for this_source_file_path in selected_source_file_paths:
			
			this_source_location,this_source_file_name=os.path.split(this_source_file_path)
		
			for target_dir_info in target_dirs_info:
			
				target_path=target_dir_info[0]
				if len(target_dir_info)==3:
				
					if not target_dir_info[1:]==[source_server_user_name,source_server_address]:
						raise ValueError('Copying between two different servers not currently supported.')
					else:	#copy on server, unless this_source_file_name already present in target_path
						execute_shell_command(['cp','-r','-n',this_source_file_path,target_path],server_username=source_server_user_name,server_address=source_server_address,wait_to_finish=True)
						
				elif len(target_dir_info)==1:
				
					if not(source_server_address):	#copy locally, unless this_source_file_name already present in target_path
						execute_shell_command(['cp','-r','-n',this_source_file_path,target_path],wait_to_finish=True)
					else:	#download from server, unless this_source_file_name already present in target_path
						execute_shell_command(['rsync','--ignore-existing',server_username+'@'+server_address+':'+this_source_file_path, target_path],wait_to_finish=True)



def execute_shell_command(shell_command_arguments,server_username='',server_address='',collect_response=True,wait_to_finish=False):
	"""Execute a shell command. If server_username and server_address are defined, then execute it remotely over ssh; otherwise locally. If run on the server
		then assumes a public/private key pair has been set up, as described for instance here:
		https://www.pragmaticlinux.com/2021/05/configure-ssh-for-login-without-a-password/.
	
	Arguments:
		shell_command_arguments (list of strings). Command and arguments that define the shell call. So basically what one would type in the shell but 
			split into separate strings at the spaces. This variable is passed as such to subprocess.Popen() if the command is run locally. In that case subprocess.Popen()
			is smart about concatenating the individual elements together into a large command while paying attention to spaces and quotes and such within the
			individual elements (escaping spaces, for instance). If the command is run on a server, then the elements are naively joined by spaces. This may run into trouble
			when trying to run commands with, for instance, a directory name that has a space in it, on the server.
		server_username (string, optional). If defined, then this is the username for the server.
		server_address (string, optional). If defined, then this is the server address. If both this and server_username are defined, then command will be run on server;
			otherwise locally.
		collect_response (bool, optional). If False, then terminal response will not be converted to list of strings and will not be returned.
		wait_to_finish (bool, optional). If True, then python will wait for this process to finish before continuing. For now only works for local processes. If False (default
			then will try to continue after command.
		
	Returns:
		response_strings (list of strings, or None). Each entry is one line of whatever would be printed in the shell when running the command defined by 
		shell_command_arguments. None is returned if collect_response is set to False.
	
	To do:
		Currently wait_to_finish is ignored for processes that are run on the server. I think ssh_client.exec_command() may wait by default, but unsure.
	"""
	
	general_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	if server_username and server_address:
		
		ssh_client = paramiko.SSHClient()
		ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())	#to avoid [server_address] 'not found in known_host'
		ssh_client.connect(server_address,22,server_username)
		
		stdin, stdout, stderror = ssh_client.exec_command(' '.join(shell_command_arguments))
		if collect_response:
			response_strings=[line.strip() for line in stdout.readlines()]
		else:
			response_strings=None
		
		ssh_client.close()
		
	else:
		
		process=subprocess.Popen(shell_command_arguments, stdout=subprocess.PIPE)
		if wait_to_finish:
			process.wait()
		
		if collect_response:
			response_strings=[(line.strip()).decode("utf-8")  for line in process.stdout.readlines()]
		else:
			response_strings=None

	return response_strings



def interactively_select_strings(strings_to_choose_from,prompt_string='Which ones shall we choose?'):
	
	"""Provide the user an opportunity to interactively select certain strings from among candidate strings provided to the function. Oftentimes
		these strings are paths, but any strings will do.
	
	Arguments:
		strings_to_choose_from (list of strings). Candidate strings available. Can be absolute paths, relative paths, filenames, or other strings.
		prompt_string (string, optional). First part of the prompt that will be shown to the user (final part is generic instruction), specifying
			what kind of selection this is.
	
	Returns:
		selected_strings (list of strings). List of all strings that have made the selection. Will correspond to strings_to_choose_from in terms of
		 	whether these will be absolute paths, relative paths, filenames, or other strings.
	"""
	
	general_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	print(prompt_string+'\nIf entering multiple indices, then separate them by commas. To select everything, please press enter. To select nothing, please enter -1.\n')
	for string_index,the_string in enumerate(strings_to_choose_from):
		print(str(string_index)+'\t'+the_string+'\t'+str(string_index))
	user_input = input('\n')

	if user_input=='':
		indices=range(len(strings_to_choose_from))
	elif user_input=='-1':
		indices=[]
	else:
		try:
			indices=[int(index_string) for index_string in user_input.split(',')]
		except:
			raise ValueError('You need to enter indices separated by commas, or press enter.')
			
	selected_strings=[strings_to_choose_from[index] for index in indices]
	
	return selected_strings



def unpack_tar_gz_archives(directory):
	
	"""Unpack all tar.gz archives in the directory provided by the argument 'directory', but skip if a folder with the same name as a given
		tar.gz archive is already present in that directory.
	
	Arguments:
		directory (string). Absolute path of directory within which archives can be found. Unpacked result will also be put in same directory.
	
	Returns:
		Nothing.
	
	To do:
		Consider adding the option of removing the original tar.gz after unpacking to a folder.
	"""
	
	general_tools_logger.info('Running '+inspect.stack()[0][3])	#inspect.stack()[0][3] is the name of the current function
	
	all_existing_subdirs=[os.path.split(candidate)[1] for candidate in glob.glob(os.path.join(directory,'*')) if os.path.isdir(candidate)]
	for tar_gz_fullpath in glob.glob(os.path.join(directory,'*.tar.gz')):
		directory,filename=os.path.split(tar_gz_fullpath)
		unpacked_subdir_name=filename.split('.')[0]
	
		if not unpacked_subdir_name in all_existing_subdirs:
			os.chdir(directory)
			execute_shell_command(['tar','-xf',tar_gz_fullpath],collect_response=False,wait_to_finish=True)
		else:
			general_tools_logger.info('Skipping unpacking operation for '+tar_gz_fullpath+' because unpacked data already present')