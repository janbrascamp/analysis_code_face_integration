function x0 = initial(obj)
% Returns initial guess for the model parameters
%
% Syntax:
%   x0 = obj.initial()
%
% Description:
%   Initial values for model
%
% Inputs:
%   none
%
% Optional key/value pairs:
%   none
%
% Outputs:
%   x0                    - 1xnParams vector.
%


% Obj variables
typicalGain = obj.typicalGain;
nParams = obj.nParams;
nGainParams = obj.nGainParams;
nAdaptParams = obj.nAdaptParams;

% Assign the x0 variable
x0 = zeros(1,nParams);

% Initialize the model with the gain parameters at the typicalGain
x0(1:nGainParams) = typicalGain;

% set the mu parameter to 0, which is the pure adaptation case
x0(nGainParams+1) = 0;
x0(nGainParams+2) = typicalGain;

% x0 HRF: Flobs population mean amplitudes
x0(nGainParams+nAdaptParams+1:nParams) = [0.86, 0.09, 0.01];


end

