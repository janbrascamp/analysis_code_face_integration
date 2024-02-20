% run_mattar_analysis_on_MSU_face_data.m
%
% Adapted from demoMattarAdapt.m provided by Geoff Aguirre.
% Intended to be run on the Circ server of MSU Psychology, to apply to the Thakkar Lab's face adaptation dataset the Aguirre lab's analysis described in Mattar et al. Curr Biol 26, 1669–1676 (2016).
%   
% To run: in an SSH terminal to the Circ server, navigate to the folder where this script is located, and run it like this: /usr/local/bin/matlab -nodesktop -nosplash -batch run_mattar_analysis_on_MSU_face_data.
% Or, if you like: nohup /usr/local/bin/matlab -nodesktop -nosplash -batch run_mattar_analysis_on_MSU_face_data &> nohupout.out &
%

% Housekeeping
clear
close all
	
% add dependencies to path
addpath(genpath('/home/brascamp/analysis_code_face_integration/mattar_analysis/'));

% identify relevant paths and folder names
base_of_BIDS_tree='/array/fmri/PI/thakkar/Face_Int/data_pipeline_Jan_starting_April_23/';
input_path_func_data=fullfile(base_of_BIDS_tree,'derivatives','fMRIPrep');
input_path_event_data=fullfile(base_of_BIDS_tree,'rawdata');
output_folder_name='mattar_analysis';
output_data_path=fullfile(base_of_BIDS_tree,'derivatives',output_folder_name);

% Whole brain or one voxel?
fit_one_voxel = false;

% The smoothing kernel for the fMRI data in space
smooth_sd = 0.75;

% The polynomial degree used for high-pass filtering of the timeseries
poly_deg = 1;

% The TR of the fMRI data, in seconds
tr = 2.2;

% The number of unique face stimuli
nFaces = 27;

% Set the typicalGain, which is about 0.1 as we have converted the data to
% proportion change
typical_gain = 0.1;

% The IDs of the subjects whose data we want to analyze
subject_IDs={};	%enter subject IDs here (format: {'sub-A0001','sub-C0103'} or leave empty for the variable to be filled with all subject IDs identified by data_folder_contents

if length(subject_IDs)==0
	input_folder_contents=dir(input_path_func_data);
	for file_or_folder_counter = 1:length(input_folder_contents)
		this_file_or_folder=input_folder_contents(file_or_folder_counter);
		if ~isempty(strfind(this_file_or_folder.name,'sub-')) & isempty(strfind(this_file_or_folder.name,'html'))
			subject_IDs=[subject_IDs,this_file_or_folder.name];
		end
	end	
end

%perform the analysis on the data of each subject in subject_IDs in turn
for subject_index = 1:length(subject_IDs)
	
	subject_ID=string(subject_IDs(subject_index));

	% Place to save the results files
	this_output_data_path = convertStringsToChars(fullfile(output_data_path,subject_ID,'ses-1','func'));
	
	if ~isfolder(this_output_data_path)	%only continue analysis if the output folder doesn't exist yet

		mkdir(this_output_data_path);
	
		% Paths and filenames for the input data: provide the paths and collect all files there that have the right name pattern
		% First the functional data, then the event data
	
		this_input_path_func_data = convertStringsToChars(fullfile(input_path_func_data,subject_ID,'ses-1','func'));
		this_input_path_event_data = convertStringsToChars(fullfile(input_path_event_data,subject_ID,'ses-1','func'));
	
		these_input_paths={this_input_path_func_data, this_input_path_event_data};
		file_name_search_strings={'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz','_events.tsv'};	%strings to search for within file names to see whether file in question is input file we're interested in
	
		for func_event_toggle=1:2
	
			this_input_path=string(these_input_paths(func_event_toggle));
			this_file_name_search_string=string(file_name_search_strings(func_event_toggle));
		
			input_folder_contents=dir(this_input_path);
	
			file_names={};
	
			for file_or_folder_counter = 1:length(input_folder_contents)
				this_file_or_folder=input_folder_contents(file_or_folder_counter);
				if ~isempty(strfind(this_file_or_folder.name,this_file_name_search_string)) & isempty(strfind(this_file_or_folder.name,'._sub-'))
					file_names=[file_names,this_file_or_folder.name];
				end
			end
		
			if func_event_toggle==1
				func_file_names=sort(file_names);
			else
				event_file_names=sort(file_names);
			end
		end
	
		% Get the stimulus files from the event file content.
		[stimulus,stimTime] = parseEventFiles(this_input_path_event_data,event_file_names);
	
		% Get the functional data files
		[data,templateImage] = parseDataFiles(this_input_path_func_data,func_file_names,smooth_sd);
	
		% Pick the voxels to analyze
		xyz = templateImage.volsize;
		if fit_one_voxel
		    % A single voxel that is in the right FFA
		    vxs = 83793;
		    averageVoxels = true;
		else
		    % Create a mask of brain voxels
		    brainThresh = 2000;
		    vxs = find(reshape(templateImage.vol, [prod(xyz), 1]) > brainThresh);
		    averageVoxels = false;
		end

		% Create the model opts, which includes stimLabels and typicalGain. The
		% paraSD key-value controls how varied the HRF solutions can be. A value of
		% 3 is fairly conservative and will keep the HRFs close to a canonical
		% shape. This is necessary for the current experiment as the stimulus
		% sequence does not uniquely constrain the temporal delay in the HRF.
		stimLabels = cellfun(@(x) sprintf('face_%02d',str2double(string(x))),num2cell(1:nFaces),'UniformOutput',false);
		stimLabels = [stimLabels,'firstFace','repeatFace','right-left'];
		modelOpts = {'stimLabels',stimLabels,'typicalGain',typical_gain,'paraSD',3,'polyDeg',poly_deg};

		% Define the modelClass
		modelClass = 'mattarAdapt';

		% Call the forwardModel
		results = forwardModel(data,stimulus,tr,...
		    'stimTime',stimTime,...
		    'vxs',vxs,...
		    'averageVoxels',averageVoxels,...
		    'verbose',true,...
		    'modelClass',modelClass,...
		    'modelOpts',modelOpts);

		% Show the results figures
		figFields = fieldnames(results.figures);
		if ~isempty(figFields)
		    for ii = 1:length(figFields)
		        figHandle = struct2handle(results.figures.(figFields{ii}).hgS_070000,0,'convert');
		        figHandle.Visible = 'on';
		    end
		end

		% Save some files if we processed the whole brain
		if ~fit_one_voxel
			file_name_stem=[convertStringsToChars(subject_ID) '_task-main_space-MNI152NLin2009cAsym_desc-']
		
		    % Save the results
		    fileName = fullfile(this_output_data_path,[file_name_stem 'mattarAdaptResults.mat']);
		    save(fileName,'results');

		    % Save the template image
		    fileName = fullfile(this_output_data_path,[file_name_stem 'epiTemplate.nii']);
		    MRIwrite(templateImage, fileName);

		    % Save a map of R2 values
		    newImage = templateImage;
		    volVec = results.R2;
		    volVec(isnan(volVec)) = 0;
		    newImage.vol = reshape(volVec,xyz(1),xyz(2),xyz(3));
		    fileName = fullfile(this_output_data_path,[file_name_stem 'mattarAdapt_R2.nii']);
		    MRIwrite(newImage, fileName);

		    % Save maps for the various param vals
		    paramLabels = {'firstFace','repeatFace','right-left','adaptMu','adaptGain'};
		    paramIdx = nFaces+1:nFaces+length(paramLabels);
		    for ii = 1:length(paramLabels)
		        newImage = templateImage;
		        volVec = results.params(:,paramIdx(ii));
		        volVec(isnan(volVec)) = 0;
		        newImage.vol = reshape(volVec,xyz(1),xyz(2),xyz(3));
		        fileName = fullfile(this_output_data_path,[file_name_stem '_mattarAdapt_' paramLabels{ii} '.nii']);
		        MRIwrite(newImage, fileName);
		    end

		end
	end
end
