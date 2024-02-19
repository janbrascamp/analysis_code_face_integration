
function nameOut = escapeFileCharacters(nameIn)
% Sanitize file strings to be used in system commands

nameOut = strrep(nameIn,' ','\ ');
nameOut = strrep(nameOut,'(','\(');
nameOut = strrep(nameOut,')','\)');
end