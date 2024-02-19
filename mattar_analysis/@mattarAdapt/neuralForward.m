function signal = neuralForward(obj, x)
% Neural forward model. Provides the predicted neural signal for a set of
% parameters
%
% Syntax:
%   signal = obj.neuralForward(x)
%
% Description:
%   Returns a time-series vector that is the predicted neural response to
%   the stimulus, based upon the parameters provided in x.
%
% Inputs:
%   x                     - 1xnParams vector.
%
% Optional key/value pairs:
%   none
%
% Outputs:
%   signal                   - 1xtime vector.
%
% Obj variables
stimulus = obj.stimulus;
nGainParams = obj.nGainParams;
stimAcqGroups = obj.stimAcqGroups;

% How many acquisitions do we have?
nAcq = max(stimAcqGroups);

% Get the adaptation parameters
mu = x(nGainParams+1);
adaptGain = x(nGainParams+2);

% Create the signal based upon the gain parameters
gainSignal = stimulus(:,1:nGainParams)*x(1:nGainParams)';

% Initialize an adapt response vector
adaptSignal = nan(size(gainSignal));

% Now loop through the trials in each acquisition and create the integrated
% adaptation effect based upon the drifting prior
for aa = 1:nAcq

    % Set the prior to the center of the stimulus space
    r = [0,0,0];

    % Get the stimulus indices for this acquisition
    idx = find(stimAcqGroups == aa);

    % Loop through the events in this acquisition and calculate the
    % response vector
    for ii = 1:length(idx)

        % Get the location of the current stimulus in face space
        s = stimulus(idx(ii),nGainParams+1:nGainParams+3);

        % Check that we have a defined position in face space
        if ~any(isnan(s))

            % The response to this stimulus is proportional to the L2 normal of the distance of
            % the current stimulus from the drifting prior
            adaptSignal(idx(ii)) = norm(s-r,2);

            % Update the prior according to mu
            r = r + (1-mu) * (s - r);

        end

    end
end

% Mean center the adaptSignal and set the remainder of the vector to zero
idx = ~isnan(adaptSignal);
adaptSignal(idx) = adaptSignal(idx)-mean(adaptSignal(idx));
idx = isnan(adaptSignal);
adaptSignal(idx) = 0;

% Apply the adaptGain
adaptSignal = adaptSignal * adaptGain;

% Add the adaptSignal to the gainSignal
signal = gainSignal + adaptSignal;

end

