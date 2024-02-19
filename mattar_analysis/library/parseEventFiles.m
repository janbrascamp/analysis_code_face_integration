function [stimulus,stimTime] = parseEventFiles(rawEventPath,eventFileNames)
% Creates the stimulus variable for forwardModel based upon event files
%
% This needs to be updated to take inputs that dynamicallty define the
% event file location. This is currently all hard-coded.


% The number of events per eventFile
nAcqs = length(eventFileNames);
nEvents = 147;

% The number of gain params is the number of unique faces, plus vectors for
% first face in the block, perfect repetitions, and right vs left button
% presses for the cover task
nUniqueFaces = 27;
nGainParams = 27 + 3;

% The total number of stimulus rows is nGainParams plus the three vectors
% that describe position in face space.
nStimMatRows = nGainParams + 3;

% How much to pad before and after the events for the blank periods
preStimTimeSecsRaw = 30;
postStimTimeSecsRaw = 33;

% A pattern used to find the face identity index within the stimulus name
pat = digitsPattern(1,2);

% Grab the temporal support from these files and use this to figure out the
% deltaT for the stimuli, and the start time of stimulus presentation
for ee = 1:length(eventFileNames)

    % Get this eventTable
    fileName = fullfile(rawEventPath,eventFileNames{ee});
    eventStruct = tdfread(fileName);

    % Get the raw event times
    if ee == length(eventFileNames)
        % Handle the special case of the last acquisition, that had one fewer
        % events
        rawTime{ee} = eventStruct.onset(1:nEvents-1);
    else
        rawTime{ee} = eventStruct.onset(1:nEvents);
    end
end
stimDeltaT = round(mean(cellfun(@(x) mean(diff(x)),rawTime)),8);

% Calculate the time to model prior to the first event and after the last
% event
nEventsPre = ceil(preStimTimeSecsRaw/stimDeltaT);
preStimTimeSecs = nEventsPre*stimDeltaT;
nEventsPost = ceil(postStimTimeSecsRaw/stimDeltaT);
postStimTimeSecs = nEventsPost*stimDeltaT;

% Clear the stimulus cell array variable
stimulus = [];

% loop over event files
for ee = 1:nAcqs

    % Get this eventTable
    fileName = fullfile(rawEventPath,eventFileNames{ee});
    eventStruct = tdfread(fileName);

    % Get the raw event times. We just care about the timing of the first
    % stimulus
    firstEventTime = eventStruct.onset(1);

    % Create and store the stimTime
    stimTime{ee} = (firstEventTime - preStimTimeSecs:stimDeltaT:firstEventTime+(nEvents-1)*stimDeltaT+postStimTimeSecs);

    % Initialize the stimMat. This includes filling the face space position
    % vectors with nans
    stimMat = zeros(nStimMatRows,nEventsPre+nEvents+nEventsPost);
    stimMat(nUniqueFaces+2:nUniqueFaces+4,:) = nan;

    % We detect perfect repetitions of faces and model them with their
    % own, separate covariate
    faceIDLast = 0;

    % Loop through the events and generate the stimMat
    for ii = 1:(nEvents-(ee==nAcqs))

        % Identify this face stim
        str = extract(eventStruct.stim_file(ii,:),pat);
        faceID = str2double(str{1});

        % Enter a delta function to model the gain for this stimulus
        stimMat(faceID,ii+nEventsPre) = 1;

        % If this is the first face of an acquisition, model that
        if ii == 1
        stimMat(nUniqueFaces+1,ii+nEventsPre) = 1;
        else
        stimMat(nUniqueFaces+1,ii+nEventsPre) = -1;
        end

        % Detect if this is a perfect face repetition
        if faceID == faceIDLast
           stimMat(nUniqueFaces+2,ii+nEventsPre) = 1;
        else
           stimMat(nUniqueFaces+2,ii+nEventsPre) = -1;
        end
        
        % Model if the button press was right or left sided
        if contains(eventStruct.dot_side(ii,:),'Right')
            stimMat(nUniqueFaces+3,ii+nEventsPre) = 1;
        end
        if contains(eventStruct.dot_side(ii,:),'Left')
            stimMat(nUniqueFaces+3,ii+nEventsPre) = -1;
        end

        % Get the [x,y,z] location in face space
        stimMat(nUniqueFaces+4,ii+nEventsPre) = str2double(eventStruct.face_x_loc_in_similarity_space_HC(ii,:));
        stimMat(nUniqueFaces+5,ii+nEventsPre) = str2double(eventStruct.face_y_loc_in_similarity_space_HC(ii,:));
        stimMat(nUniqueFaces+6,ii+nEventsPre) = str2double(eventStruct.face_z_loc_in_similarity_space_HC(ii,:));

        % Update the faceIDLast
        faceIDLast = faceID;

    end

    % Mean center the unique face rows
    for ff = 1:nUniqueFaces
        stimMat(ff,:) = stimMat(ff,:) - mean(stimMat(ff,:));
    end

    % Mean center the other gain rows by adjusting the magnitude of the
    % negative arm to balance the positive arm
    for ff = nUniqueFaces+1:nUniqueFaces+3
        vec = stimMat(ff,:);
        posIdx = vec>0;
        negIdx = vec<0;
        posSum = sum(vec(posIdx));
        negSum = sum(vec(negIdx));
        negVal = posSum/negSum;
        vec(negIdx) = negVal;
        vec(isnan(vec)) = 0;
        stimMat(ff,:) = vec;
    end

    % Store the stimMat
    stimulus{ee} = stimMat;

end


end