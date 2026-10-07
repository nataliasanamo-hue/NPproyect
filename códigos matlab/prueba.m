% export_propRipS.m
% Exports rippleAnalysis.propRipS (a MATLAB table, which Python cannot read) of every
% session to a plain CSV.
%
% READS  : <session>\analyset\rippleAnalysis.mat   (only reads it, never changes it)
% WRITES : <session>\ProcesamientoNatalia\propRipS.csv
%
% Run it once in MATLAB. At the end it shows the column names of the table:
% please copy that line and send it.

baseDir = 'D:\Violeta\ANALISIS';
outName = 'ProcesamientoNatalia';

animalDirs = dir(baseDir);
animalDirs = animalDirs([animalDirs.isdir] & ~ismember({animalDirs.name},{'.','..'}));

nOK = 0; nFail = 0; lastNames = {};

for iA = 1:numel(animalDirs)
    animalPath = fullfile(baseDir, animalDirs(iA).name);
    sessionDirs = dir(animalPath);
    sessionDirs = sessionDirs([sessionDirs.isdir] & ~ismember({sessionDirs.name},{'.','..'}));

    for iS = 1:numel(sessionDirs)
        sessionPath = fullfile(animalPath, sessionDirs(iS).name);
        ripFile = fullfile(sessionPath, 'analyset', 'rippleAnalysis.mat');
        if ~isfile(ripFile), continue; end

        try
            S = load(ripFile, 'rippleAnalysis');
            P = S.rippleAnalysis.propRipS;
            if isstruct(P), P = struct2table(P); end   % in case it is a struct
            if ~istable(P), P = array2table(P); end    % in case it is a plain matrix

            outDir = fullfile(sessionPath, outName);
            if ~isfolder(outDir), mkdir(outDir); end
            writetable(P, fullfile(outDir, 'propRipS.csv'));

            lastNames = P.Properties.VariableNames;
            nOK = nOK + 1;
        catch ME
            warning('Error in %s: %s', sessionPath, ME.message);
            nFail = nFail + 1;
        end
    end
end

fprintf('\nExported %d sessions, %d failed.\n', nOK, nFail);
fprintf('Columns of propRipS: %s\n', strjoin(lastNames, ', '));