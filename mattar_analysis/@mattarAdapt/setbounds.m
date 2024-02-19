function setbounds(obj)
% Sets the bounds on the model parameters
%
% Syntax:
%   obj.setbounds()
%
% Description:
%   Bounds for the model
%
%   These are specified as 1 x nParams vectors.
%
% Inputs:
%   none
%
% Optional key/value pairs:
%   none
%
% Outputs:
%   none
%

% Obj variables
nParams = obj.nParams;
nGainParams = obj.nGainParams;
nAdaptParams = obj.nAdaptParams;

% Define outputs
lb = nan(1,nParams);
ub = nan(1,nParams);

% The gain parameters are unbounded
lb(1:nGainParams) = -Inf;             % gain
ub(1:nGainParams) = Inf;              % gain

% The mu parameter is bounded by zero and one. The adaptGain is unbounded.
lb(nGainParams+1) = 0;
ub(nGainParams+1) = 1;
lb(nGainParams+2) = -Inf;
ub(nGainParams+2) = Inf;

% The HRF shape parameters vary by model type
switch obj.hrfType
    case 'flobs'
        
        % Object properties associated with the FLOBS eigenvectors
        mu = obj.mu;
        C = obj.C;
        
        % Set bounds at +- n * SDs of the norm distributions of the FLOBS
        % parameters, default n = 15
        sdBound = obj.paraSD * diag(C)';
        
        lb(nGainParams+nAdaptParams+1:nParams) = mu-sdBound;	% FLOBS eigen1, 2, 3
        ub(nGainParams+nAdaptParams+1:nParams) = mu+sdBound;	% FLOBS eigen1, 2, 3

    case 'gamma'
        lb(nGainParams+nAdaptParams+1:nParams) = [2 6 0];	% Gamma1,2, and undershoot gain
        ub(nGainParams+nAdaptParams+1:nParams) = [8 12 2];	% Gamma1,2, and undershoot gain

    otherwise
        error('Not a valid hrfType')
end

% Store the bounds in the object
obj.lb = lb;
obj.ub = ub;

% Store the FiniteDifferenceStepSize for the model. See here for more
% details:
%   https://www.mathworks.com/help/optim/ug/optimization-options-reference.html
FiniteDifferenceStepSize = nan(1,nParams);
FiniteDifferenceStepSize(1,:) = sqrt(eps);
obj.FiniteDifferenceStepSize = FiniteDifferenceStepSize;

end

