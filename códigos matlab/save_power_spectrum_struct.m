

%% Put all power spectrum info into a structure, to compare controls and mutants and between animals 

clear; close all; clc; 

% directory where data are stored
working_dir = 'C:\Users\Violeta\Desktop\NPC_Project';
% go to that directory (cd = change directory)
cd(working_dir);
% get a breakdown of all files and folders inside using "dir"
files = dir(working_dir); 
files(ismember({files.name},{'.','..'})) = []; % get rid of first 2 invisible folders calles '.' and '..'
files = files([files.isdir]); % also get rid of any file that is not a folder (all my data will be in folders)

% get paths of folders to load (with data from NPC of CTR sessions)
files2load = zeros(size(files,1),1);

for iFile = 1:size(files,1)
    if contains(files(iFile).name,'CTR') || contains(files(iFile).name,'NPC') && files(iFile).isdir
        files2load(iFile) = 1;
    end
end
if any(files2load == 0)
    files(find(files2load==0)) = [];
end

% create cell array of directory paths for each session to load later
paths = cell(size(files));
for iPath = 1:size(files,1)
    % find those sessions within strcuture "files"
    paths{iPath} = fullfile(files(iPath).folder,files(iPath).name);
end

% create structure with all the info for all sessions to later access them
% easily for loading of the data you want

sessions = struct([]);
for iPath = 1:length(paths)
    iPathFiles = dir(paths{iPath});
    iPathFiles(ismember({iPathFiles.name},{'.','..'})) = []; 
    for iSess = 1:length({iPathFiles.name})
        index = size(sessions,2)+1;
        sessions(index).session = fullfile(iPathFiles(iSess).folder, iPathFiles(iSess).name);
        [~,animalName]= fileparts(paths{iPath});
        sessions(index).animal = animalName;
        if contains(animalName, 'CTR')
            sessions(index).condition = 'CTR';
        else
            sessions(index).condition = 'NPC';
        end
    end
end

clearvars -except sessions paths working_dir


%% Open session, load analyset output, get theta and LIA times, calculate pS
ctr_sessions = sessions(contains({sessions.condition},'CTR'));
mut_sessions = sessions(contains({sessions.condition},'NPC'));
sessionAnalysisCTR = struct([]);
sessionAnalysisMUT = struct([]);
% === Seleccionar archivo Excel manualmente ===
[filename, pathname] = uigetfile({'*.xlsx;*.xls;*.csv','Excel Files (*.xlsx, *.xls, *.csv)'}, ...
    'Selecciona el archivo Excel');
if isequal(filename,0)
    error('No se seleccionó ningún archivo. El script se detuvo.');
else
    excelfile = fullfile(pathname, filename);
    wellnessScale = readtable(excelfile);
end

% CTR sessions
for iSess=1:size(ctr_sessions,2)
    sessPath = ctr_sessions(iSess).session;
    cd(sessPath);
    if isfolder('analyset')
        analysetFolder = fullfile(sessPath,'analyset');
        files = dir(fullfile(analysetFolder, '*.mat'));
        files2load = zeros(size(files,1),1);
        for iFile = 1:length(files2load)
            if contains(files(iFile).name,'ripple') || contains(files(iFile).name,'configChannels') ||...
               contains(files(iFile).name,'theta') || contains(files(iFile).name,'psProfile') 
                files2load(iFile) = 1;
            end
        end
        for fileNum = find(files2load)'
            load(fullfile(files(fileNum).folder,files(fileNum).name));
        end

        % pS windows: 5000 samples (2 sec)
        % pS window moves by: 1250 samples (0.5 sec)
        % ps matrix (psProfile.pS) =  dB value per 0.5Hz window x windows x channels
        % e.g. {1,1,1} in matrix = dB value for 1-1.5Hz bin for channel 1
        % i.e. {dB,window,channel}
        fs = 2500;
        windowSize = psProfile.params.tWin/fs; % in seconds
        windowStep = psProfile.params.dtWin/fs;
        thetaTimes = thetaLIAAnalysis.timesTheta;
        LIATimes = thetaLIAAnalysis.timesLIA;
        pSwindowsTheta = zeros(size(psProfile.pS,2),1);
        pSwindowsLIA = zeros(size(psProfile.pS,2),1);
        % create empty array to fill in with real timestamps of pS windows
        pSbins = nan(size(pSwindowsTheta,1),2);
        pSbins(:,1) = [0:windowStep:(length(pSbins)*windowStep-windowStep)]';
        pSbins(:,2) = pSbins(:,1)+windowSize;
        
        
        % THETA
        % Create a loop to check which pS windows the theta/LIA windows fall within
        for iTwindow = 1:length(thetaTimes)
            % Theta time window:
            window = thetaTimes(iTwindow,:);
            if diff(window) > windowSize + windowStep
                % disp('Condition met. Executing for loop:');
                for iTimes = 1 :length(pSbins)
                % current pSbin time window:
                    times = pSbins(iTimes,:);
                    % Check if current pSbin time window contains theta time window
                    if sum((window > times(1))) * sum((window < times(2))) > 0
                        pSwindowsTheta(iTimes) = 1;
                    end
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % LIA
        % Create a loop to check which pS windows the theta/LIA windows fall within
        for iLwindow = 1:length(LIATimes)
            % LIA time window:
            window = LIATimes(iLwindow,:);
            if diff(window) > windowSize + windowStep
                % disp('Condition met. Executing for loop:');
                for iTimes = 1 :length(pSbins)
                     % current pSbin time window:
                    times = pSbins(iTimes,:);
                    % Check if current pSbin time window contains LIA time window
                    if sum((window > times(1))) * sum((window < times(2))) > 0
                    pSwindowsLIA(iTimes) = 1;
                    end
                end
            end
        end
        
        % Make a structure to save pSbinsTheta/pSTheta and pSbinsLIA/pSLIA for each session
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% session %%%% meaned pSTheta %%%% meaned pSLIA %%%%% rippleAnalysis.propRipS %%% channelInfo %%%  wellnessScale
        sessionAnalysisCTR(iSess).session = sessPath;
        sessionAnalysisCTR(iSess).animal = ctr_sessions(iSess).animal;
        pSThetaSubset = psProfile.pS(:,logical(pSwindowsTheta),:);
        pSLIASubset = psProfile.pS(:,logical(pSwindowsLIA),:);
        sessionAnalysisCTR(iSess).pSTheta = squeeze(mean(pSThetaSubset,2));
        sessionAnalysisCTR(iSess).thetaBins = sum(pSwindowsTheta);
        sessionAnalysisCTR(iSess).LIABins = sum(pSwindowsLIA);
        sessionAnalysisCTR(iSess).pSLIA = squeeze(mean(pSLIASubset,2));
        sessionAnalysisCTR(iSess).ripples = rippleAnalysis.propRipS;
        sessionAnalysisCTR(iSess).realChArea = configChannels.realChArea;
        sessionAnalysisCTR(iSess).channelInfo = configChannels.ch;
        sessionAnalysisCTR(iSess).wellnessScale = wellnessScale(contains(wellnessScale.Animal, ctr_sessions(iSess).animal),:);
    end
end

        
%% Same for MUT animals

% MUT sessions
for iSess=1:size(mut_sessions,2)
    sessPath = mut_sessions(iSess).session;
    cd(sessPath);
    if isfolder('analyset')
        analysetFolder = fullfile(sessPath,'analyset');
        files = dir(fullfile(analysetFolder, '*.mat'));
        files2load = zeros(size(files,1),1);
        for iFile = 1:length(files2load)
            if contains(files(iFile).name,'ripple') || contains(files(iFile).name,'configChannels') ||...
               contains(files(iFile).name,'theta') || contains(files(iFile).name,'psProfile') 
                files2load(iFile) = 1;
            end
        end
        for fileNum = find(files2load)'
            load(fullfile(files(fileNum).folder,files(fileNum).name));
        end

        % pS windows: 5000 samples (2 sec)
        % pS window moves by: 1250 samples (0.5 sec)
        % ps matrix (psProfile.pS) =  dB value per 0.5Hz window x windows x channels
        % e.g. {1,1,1} in matrix = dB value for 1-1.5Hz bin for channel 1
        % i.e. {dB,window,channel}
        fs = 2500;
        windowSize = psProfile.params.tWin/fs; % in seconds
        windowStep = psProfile.params.dtWin/fs;
        thetaTimes = thetaLIAAnalysis.timesTheta;
        LIATimes = thetaLIAAnalysis.timesLIA;
        pSwindowsTheta = zeros(size(psProfile.pS,2),1);
        pSwindowsLIA = zeros(size(psProfile.pS,2),1);
        % create empty array to fill in with real timestamps of pS windows
        pSbins = nan(size(pSwindowsTheta,1),2);
        pSbins(:,1) = [0:windowStep:(length(pSbins)*windowStep-windowStep)]';
        pSbins(:,2) = pSbins(:,1)+windowSize;
        
        
        % THETA
        % Create a loop to check which pS windows the theta/LIA windows fall within
        for iTwindow = 1:length(thetaTimes)
            % Theta time window:
            window = thetaTimes(iTwindow,:);
            if diff(window) > windowSize + windowStep
                % disp('Condition met. Executing for loop:');
                for iTimes = 1 :length(pSbins)
                % current pSbin time window:
                    times = pSbins(iTimes,:);
                    % Check if current pSbin time window contains theta time window
                    if sum((window > times(1))) * sum((window < times(2))) > 0
                        pSwindowsTheta(iTimes) = 1;
                    end
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % LIA
        % Create a loop to check which pS windows the theta/LIA windows fall within
        for iLwindow = 1:length(LIATimes)
            % LIA time window:
            window = LIATimes(iLwindow,:);
            if diff(window) > windowSize + windowStep
                % disp('Condition met. Executing for loop:');
                for iTimes = 1 :length(pSbins)
                     % current pSbin time window:
                    times = pSbins(iTimes,:);
                    % Check if current pSbin time window contains LIA time window
                    if sum((window > times(1))) * sum((window < times(2))) > 0
                    pSwindowsLIA(iTimes) = 1;
                    end
                end
            end
        end
        
        % Make a structure to save pSbinsTheta/pSTheta and pSbinsLIA/pSLIA for each session
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% session %%%% meaned pSTheta %%%% meaned pSLIA %%%% rippleAnalysis.propRipS %%% channelInfo %%%
        sessionAnalysisMUT(iSess).session = sessPath;
        sessionAnalysisMUT(iSess).animal = mut_sessions(iSess).animal;
        pSThetaSubset = psProfile.pS(:,logical(pSwindowsTheta),:);
        pSLIASubset = psProfile.pS(:,logical(pSwindowsLIA),:);
        sessionAnalysisMUT(iSess).pSTheta = squeeze(mean(pSThetaSubset,2));
        sessionAnalysisMUT(iSess).thetaBins = sum(pSwindowsTheta);
        sessionAnalysisMUT(iSess).LIABins = sum(pSwindowsLIA);
        sessionAnalysisMUT(iSess).pSLIA = squeeze(mean(pSLIASubset,2));
        sessionAnalysisMUT(iSess).ripples = rippleAnalysis.propRipS;
        sessionAnalysisMUT(iSess).realChArea = configChannels.realChArea;
        sessionAnalysisMUT(iSess).channelInfo = configChannels.ch;
        sessionAnalysisMUT(iSess).wellnessScale = wellnessScale(contains(wellnessScale.Animal, mut_sessions(iSess).animal),:);
    end
end

clearvars -except sessionAnalysisMUT sessionAnalysisCTR ctr_sessions mut_sessions working_dir
%% Create structures of power spectrums for CTR and NPC during Theta and LIA
psFreqs = [1:0.5:1000];

% CTR Theta
pSThetaCTR = {sessionAnalysisCTR(:).pSTheta};
channelInfoCTR = {sessionAnalysisCTR(:).channelInfo};
chPyrCTR = nan(length(channelInfoCTR),1);
chSlmCTR = nan(length(channelInfoCTR),1);
psThetaPyrCTR = nan(size(pSThetaCTR{1},1),length(channelInfoCTR));
psThetaSlmCTR = nan(size(pSThetaCTR{1},1),length(channelInfoCTR));

for iSess = 1:length(channelInfoCTR)
    if ~isempty(channelInfoCTR{iSess})
        iPyr = channelInfoCTR{iSess}.pyr;
        iSlm = channelInfoCTR{iSess}.theta;
        chPyrCTR(iSess) = iPyr;
        chSlmCTR(iSess) = iSlm;
        psThetaPyrCTR(:,iSess) = pSThetaCTR{iSess}(:,iPyr);
        psThetaSlmCTR(:,iSess) = pSThetaCTR{iSess}(:,iSlm);
    end
end


% CTR LIA

pSLIACTR = {sessionAnalysisCTR(:).pSLIA};
psLIAPyrCTR = nan(size(pSLIACTR{1},1),length(channelInfoCTR));
psLIASlmCTR = nan(size(pSLIACTR{1},1),length(channelInfoCTR));

for iSess = 1:length(channelInfoCTR)
    if ~isempty(channelInfoCTR{iSess})
        iPyr = channelInfoCTR{iSess}.pyr;
        iSlm = channelInfoCTR{iSess}.theta;
        chPyrCTR(iSess) = iPyr;
        chSlmCTR(iSess) = iSlm;
        psLIAPyrCTR(:,iSess) = pSLIACTR{iSess}(:,iPyr);
        psLIASlmCTR(:,iSess) = pSLIACTR{iSess}(:,iSlm);
    end
end

% MUT Theta
pSThetaMUT = {sessionAnalysisMUT(:).pSTheta};
channelInfoMUT = {sessionAnalysisMUT(:).channelInfo};
chPyrMUT = nan(length(channelInfoMUT),1);
chSlmMUT = nan(length(channelInfoMUT),1);
psThetaPyrMUT = nan(size(pSThetaMUT{1},1),length(channelInfoMUT));
psThetaSlmMUT = nan(size(pSThetaMUT{1},1),length(channelInfoMUT));

for iSess = 1:length(channelInfoMUT)
    if ~isempty(channelInfoMUT{iSess})
        iPyr = channelInfoMUT{iSess}.pyr;
        iSlm = channelInfoMUT{iSess}.theta;
        chPyrMUT(iSess) = iPyr;
        chSlmMUT(iSess) = iSlm;
        psThetaPyrMUT(:,iSess) = pSThetaMUT{iSess}(:,iPyr);
        psThetaSlmMUT(:,iSess) = pSThetaMUT{iSess}(:,iSlm);
    end
end

%MUT LIA

pSLIAMUT = {sessionAnalysisMUT(:).pSLIA};
psLIAPyrMUT = nan(size(pSLIAMUT{1},1),length(channelInfoMUT));
psLIASlmMUT = nan(size(pSLIAMUT{1},1),length(channelInfoMUT));

for iSess = 1:length(channelInfoMUT)
    if ~isempty(channelInfoMUT{iSess})
        iPyr = channelInfoMUT{iSess}.pyr;
        iSlm = channelInfoMUT{iSess}.theta;
        chPyrMUT(iSess) = iPyr;
        chSlmMUT(iSess) = iSlm;
        psLIAPyrMUT(:,iSess) = pSLIAMUT{iSess}(:,iPyr);
        psLIASlmMUT(:,iSess) = pSLIAMUT{iSess}(:,iSlm);
    end
end

%% Plot Figure


color_ctr = [117 117 117]/256;
color_mut = [106 27 154]/256;

% PYR pS in Theta
fig = figure;
subplot(2,2,1)
plotFill(psFreqs, psThetaPyrCTR, 'color', color_ctr); hold on;
plotFill(psFreqs, psThetaPyrMUT,  'color', color_mut); hold on;
xlim([0 80]);
legend({'','CTR','','MUT'})
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Power spectrum in PYR channel - Theta')
yscale log

% SLM pS in Theta
subplot(2,2,2)
plotFill(psFreqs, psThetaSlmCTR, 'color', color_ctr); hold on;
plotFill(psFreqs, psThetaSlmMUT, 'color', color_mut); hold on;
xlim([0 80]);
legend({'','CTR','','MUT'})
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Power spectrum in SLM channel - Theta')
yscale log

% PYR pS in LIA
subplot(2,2,3)
plotFill(psFreqs, psLIAPyrCTR, 'color', color_ctr, 'LineStyle', '--'); hold on;
plotFill(psFreqs, psLIAPyrMUT,  'color', color_mut, 'LineStyle', '--'); hold on;
xlim([0 80]);
legend({'','CTR','','MUT'})
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Power spectrum in PYR channel - LIA')
yscale log

% SLM pS in LIA
subplot(2,2,4)
plotFill(psFreqs, psLIASlmCTR, 'color', color_ctr, 'LineStyle', '--'); hold on;
plotFill(psFreqs, psLIASlmMUT,  'color', color_mut, 'LineStyle', '--'); hold on;
xlim([0 80]);
legend({'','CTR','','MUT'})
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Power spectrum in SLM channel - LIA')
yscale log

saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% STATS

deltaRange = [0 3];
deltaIndices = find((psFreqs >= deltaRange(1)).* (psFreqs <= deltaRange(2)));

alphaRange = [9 16];
alphaIndices = find((psFreqs >= alphaRange(1)).* (psFreqs <= alphaRange(2)));

thetaRange = [4 12];
thetaIndices = find((psFreqs >= thetaRange(1)).* (psFreqs <= thetaRange(2)));

slowGammaRange = [20 40];
slowGammaIndices = find((psFreqs >= slowGammaRange(1)).* (psFreqs <= slowGammaRange(2)));

midGammaRange = [40 60];
midGammaIndices = find((psFreqs >= midGammaRange(1)).* (psFreqs <= midGammaRange(2)));

fastGammaRange = [60 80];
fastGammaIndices = find((psFreqs >= fastGammaRange(1)).* (psFreqs <= fastGammaRange(2)));

%% AUTOMATION

listOfConditionTitles = {'Delta power in SLM - Theta',...
                    'Delta power in PYR - Theta',...
                    'Delta power in SLM - LIA',...
                    'Delta power in PYR - Theta',...
                    ....};
listOfIndices = {'deltaIndices',...
                 'thetaIndices',...   
                 };

listOfConditionsCTR = {};               


%% Delta power in SLM in theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Delta power in SLM - Theta';
conditionCTR = psThetaSlmCTR;
conditionMUT = psThetaSlmMUT;
pSindices = deltaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));


%% Delta power in SLM in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Delta power in SLM - LIA';
conditionCTR = psLIASlmCTR;
conditionMUT = psLIASlmMUT;
pSindices = deltaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Delta power in PYR in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Delta power in PYR - Theta';
conditionCTR = psThetaPyrCTR;
conditionMUT = psThetaPyrMUT;
pSindices = deltaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));


%% Delta power in PYR in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Delta power in PYR - LIA';
conditionCTR = psLIAPyrCTR;
conditionMUT = psLIAPyrMUT;
pSindices = deltaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [H,P,CI,STATS] = ttest2(max(x1),max(x2));
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));


%% Theta power in SLM

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Theta power in SLM';
conditionCTR = psLIASlmCTR;
conditionMUT = psLIASlmMUT;
pSindices = thetaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));



%% Theta power in PYR

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Theta power in PYR';
conditionCTR = psLIAPyrCTR;
conditionMUT = psLIAPyrMUT;
pSindices = thetaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;


%%%%%%%%%%%%%%WellnessScale-correlation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
groupCorr(wScores, [max(x1),max(x2)],'inAxis',true, 'MarkerEdgeColor',[0/255 66/255 168/255], 'MarkerSize', 15, 'MarkerColor', 'none')
title("Max Theta PYR")
ylabel("Power (dB)")
xlabel("Wellness Scale")

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Slow Gamma in SLM in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Slow Gamma power in SLM - Theta';
conditionCTR = psThetaSlmCTR;
conditionMUT = psThetaSlmMUT;
pSindices = slowGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Slow Gamma in SLM in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Slow Gamma power in SLM - LIA';
conditionCTR = psLIASlmCTR;
conditionMUT = psLIASlmMUT;
pSindices = slowGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Slow Gamma in PYR in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Slow Gamma power in PYR - Theta';
conditionCTR = psThetaPyrCTR;
conditionMUT = psThetaPyrMUT;
pSindices = slowGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Slow Gamma in PYR in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Slow Gamma power in PYR - LIA';
conditionCTR = psLIAPyrCTR;
conditionMUT = psLIAPyrMUT;
pSindices = slowGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));


%% Mid Gamma in SLM in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Mid Gamma power in SLM - Theta';
conditionCTR = psThetaSlmCTR;
conditionMUT = psThetaSlmMUT;
pSindices = midGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Mid Gamma in SLM in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Mid Gamma power in SLM - LIA';
conditionCTR = psLIASlmCTR;
conditionMUT = psLIASlmMUT;
pSindices = midGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Mid Gamma in PYR in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Mid Gamma power in PYR - Theta';
conditionCTR = psThetaPyrCTR;
conditionMUT = psThetaPyrMUT;
pSindices = midGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Mid Gamma in PYR in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Mid Gamma power in PYR - LIA';
conditionCTR = psLIAPyrCTR;
conditionMUT = psLIAPyrMUT;
pSindices = midGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));


%% Fast Gamma in SLM in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Fast Gamma power in SLM - Theta';
conditionCTR = psThetaSlmCTR;
conditionMUT = psThetaSlmMUT;
pSindices = fastGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Fast Gamma in SLM in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Fast Gamma power in SLM - LIA';
conditionCTR = psLIASlmCTR;
conditionMUT = psLIASlmMUT;
pSindices = fastGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Fast Gamma in PYR in Theta

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Fast Gamma power in PYR - Theta';
conditionCTR = psThetaPyrCTR;
conditionMUT = psThetaPyrMUT;
pSindices = fastGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% Fast Gamma in PYR in LIA

% Stats plots - get dB values for theta range Hz ONLY
condition = 'Fast Gamma power in PYR - LIA';
conditionCTR = psLIAPyrCTR;
conditionMUT = psLIAPyrMUT;
pSindices = fastGammaIndices;

conditionDataCTR = nan(length(pSindices),length(ctr_sessions));
for iSess = 1:length(ctr_sessions)
    conditionDataCTR(:,iSess) = conditionCTR(pSindices,iSess);
end
conditionDataMUT = nan(length(pSindices),length(mut_sessions));
for iSess = 1:length(mut_sessions)
    conditionDataMUT(:,iSess) = conditionMUT(pSindices,iSess);
end

% MAX
x1 = conditionDataCTR; 
x2 = conditionDataMUT; 
xsMax = {max(x1), max(x2)};
xsMean = {mean(x1,1), mean(x2,1)};
is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsMean, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[mean(x1,1), mean(x2,1)], 'MarkerEdgeColor','k');
title('Mean')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsMax, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(size(x1,2),1)',ones(size(x2,2),1)'*2],[max(x1), max(x2)], 'MarkerEdgeColor','k');
title('Max')
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});
sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));

%% RIPPLES 

% Stats plots 
condition = 'Ripple Features';
ripplesCTR = {sessionAnalysisCTR(:).ripples};
ripplesMUT = {sessionAnalysisMUT(:).ripples};

ripplesFreqCTR = nan(length(ctr_sessions),1);
ripplesAmpCTR = nan(length(ctr_sessions),1);
ripplesPowerCTR = nan(length(ctr_sessions),1);

ripplesFreqMUT = nan(length(mut_sessions),1);
ripplesAmpMUT = nan(length(mut_sessions),1);
ripplesPowerMUT = nan(length(mut_sessions),1);


for iSess = 1:length(ctr_sessions)
    if ~isempty(ripplesCTR{iSess})
        ripplesFreqCTR(iSess) = mean(ripplesCTR{iSess}.frequency,1);
        ripplesAmpCTR(iSess) = mean(ripplesCTR{iSess}.amplitude,1);
        ripplesPowerCTR(iSess) = mean(ripplesCTR{iSess}.power,1);
    end
end
for iSess = 1:length(mut_sessions)
    if ~isempty(ripplesMUT{iSess})
        ripplesFreqMUT(iSess) = mean(ripplesMUT{iSess}.frequency,1);
        ripplesAmpMUT(iSess) = mean(ripplesMUT{iSess}.amplitude,1);
        ripplesPowerMUT(iSess) = mean(ripplesMUT{iSess}.power,1);
    end
end

xFreq1 = ripplesFreqCTR; 
xFreq2 = ripplesFreqMUT; 
xsFreq = {xFreq1, xFreq2};

xAmp1 = ripplesAmpCTR; 
xAmp2 = ripplesAmpMUT; 
xsAmp = {xAmp1, xAmp2};

xPower1 = ripplesPowerCTR; 
xPower2 = ripplesPowerMUT; 
xsPower = {xPower1, xPower2};

is_paired = false;
scat_val = 0.1;

%%%%%%%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[H,P,CI,STATS] = ttest2(...);
colors = [color_ctr; color_mut];
fig = figure(position = [123,65,305,570]);
tiledlayout
nexttile
stats = groupStats(xsFreq, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(length(xFreq1),1);ones(length(xFreq2),1)*2],[xFreq1; xFreq2], 'MarkerEdgeColor','k');
% title('Frequency')
ylabel('Frequency (Hz)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsAmp, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(length(xAmp1),1);ones(length(xAmp2),1)*2],[xAmp1; xAmp2], 'MarkerEdgeColor','k');
% title('Amplitude')
ylabel('Amplitude (uV)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

nexttile
stats = groupStats(xsPower, [], ...
        'plotType', 'boxplot', ...
        'color', colors, 'inAxis', true, 'sigStar', true);
hold on
scatter([ones(length(xPower1),1);ones(length(xPower2),1)*2],[xPower1; xPower2], 'MarkerEdgeColor','k');
ylabel('Power (dB)');
xticks([1 2]);
xticklabels({'Control','Mutant'});

sgtitle(condition);
saveas(fig, fullfile(working_dir,'RESULTS',sprintf('%s.png',condition)));





%% Correlations with Wellness Scale

dirAnalyset = 'C:\Users\Violeta\Documents\DOCTORADO\PROYECTO ELECTRO CAJAL\DATA\GitHub\Analyset';
addpath(genpath(dirAnalyset))

wellnessScores = nan(size(ctr_sessions,2)+size(mut_sessions,2),1);
ctrScores = {sessionAnalysisCTR(:).wellnessScale};
mutScores = {sessionAnalysisMUT(:).wellnessScale};

wellnessScores = horzcat(ctrScores,mutScores);
wScores = nan(size(ctr_sessions,2)+size(mut_sessions,2),1);
for iSess = 1:length(wellnessScores)
    if ~isempty(wellnessScores{iSess})
        wScores(iSess) = wellnessScores{iSess}.TotalScore;
    end
end


% Ripple Amp 

% === Separar por grupos ===
nCTR = length(ctr_sessions);

wCTR = wScoresClean(1:nCTR);
rCTR = rippleAmpsClean(1:nCTR);

wMUT = wScoresClean(nCTR+1:end);
rMUT = rippleAmpsClean(nCTR+1:end);

figure; hold on

% ---------- CTR ----------
scatter(wCTR, rCTR, 80, color_ctr, 'filled', ...
    'MarkerEdgeColor', [0 0 0]*0.3, 'LineWidth', 1.2)
pCTR = polyfit(wCTR, rCTR, 1);
xC = linspace(min(wCTR), max(wCTR), 100);
plot(xC, polyval(pCTR,xC), 'Color', color_ctr, 'LineWidth', 2);

% ---------- MUT ----------
scatter(wMUT, rMUT, 80, color_mut, 'filled', ...
    'MarkerEdgeColor', [0 0 0]*0.3, 'LineWidth', 1.2)
pMUT = polyfit(wMUT, rMUT, 1);
xM = linspace(min(wMUT), max(wMUT), 100);
plot(xM, polyval(pMUT,xM), 'Color', color_mut, 'LineWidth', 2);

% ---------- Global ----------
pAll = polyfit(wScoresClean, rippleAmpsClean, 1);
xAll = linspace(min(wScoresClean), max(wScoresClean), 100);
plot(xAll, polyval(pAll,xAll), 'k--', 'LineWidth', 2);

% ---------- Correlaciones ----------
[rC,pC] = corr(wCTR, rCTR, 'Type','Pearson');
[rM,pM] = corr(wMUT, rMUT, 'Type','Pearson');
[rAll,pAllCorr] = corr(wScoresClean, rippleAmpsClean, 'Type','Pearson');

% Texto en la gráfica
yMax = max(rippleAmpsClean);
xMin = min(wScoresClean);

text(xMin+0.3, yMax*0.95, sprintf('CTR: r=%.2f, p=%.3f', rC, pC), ...
    'Color', color_ctr, 'FontSize', 11)
text(xMin+0.3, yMax*0.88, sprintf('MUT: r=%.2f, p=%.3f', rM, pM), ...
    'Color', color_mut, 'FontSize', 11)
text(xMin+0.3, yMax*0.81, sprintf('All: r=%.2f, p=%.3f', rAll, pAllCorr), ...
    'Color', 'k', 'FontSize', 11, 'FontWeight','bold')

% ---------- Estética ----------
grid on
box on
set(gca,'Color',[0.95 0.95 0.95])
xlabel('Wellness Scale', 'FontSize', 12, 'FontWeight', 'bold')
ylabel('Ripple Amplitude (\muV)', 'FontSize', 12, 'FontWeight', 'bold')
title('Correlation: Ripple Amplitude vs Wellness Scale', 'FontSize', 14, 'FontWeight', 'bold')
legend({'CTR','CTR fit','MUT','MUT fit','Global fit'}, 'Location','northeast')



% Theta power in PYR

% === Separar por grupos ===
nCTR = length(ctr_sessions);

wCTR = wScoresClean(1:nCTR);
rCTR = rippleAmpsClean(1:nCTR);

wMUT = wScoresClean(nCTR+1:end);
rMUT = rippleAmpsClean(nCTR+1:end);

figure; hold on

% ---------- CTR ----------
scatter(wCTR, rCTR, 80, color_ctr, 'filled', ...
    'MarkerEdgeColor', [0 0 0]*0.3, 'LineWidth', 1.2)
pCTR = polyfit(wCTR, rCTR, 1);
xC = linspace(min(wCTR), max(wCTR), 100);
plot(xC, polyval(pCTR,xC), 'Color', color_ctr, 'LineWidth', 2);

% ---------- MUT ----------
scatter(wMUT, rMUT, 80, color_mut, 'filled', ...
    'MarkerEdgeColor', [0 0 0]*0.3, 'LineWidth', 1.2)
pMUT = polyfit(wMUT, rMUT, 1);
xM = linspace(min(wMUT), max(wMUT), 100);
plot(xM, polyval(pMUT,xM), 'Color', color_mut, 'LineWidth', 2);

% ---------- Global ----------
pAll = polyfit(wScoresClean, rippleAmpsClean, 1);
xAll = linspace(min(wScoresClean), max(wScoresClean), 100);
plot(xAll, polyval(pAll,xAll), 'k--', 'LineWidth', 2);

% ---------- Correlaciones ----------
[rC,pC] = corr(wCTR, rCTR, 'Type','Pearson');
[rM,pM] = corr(wMUT, rMUT, 'Type','Pearson');
[rAll,pAllCorr] = corr(wScoresClean, rippleAmpsClean, 'Type','Pearson');

% Texto en la gráfica
yMax = max(rippleAmpsClean);
xMin = min(wScoresClean);

text(xMin+0.3, yMax*0.95, sprintf('CTR: r=%.2f, p=%.3f', rC, pC), ...
    'Color', color_ctr, 'FontSize', 11)
text(xMin+0.3, yMax*0.88, sprintf('MUT: r=%.2f, p=%.3f', rM, pM), ...
    'Color', color_mut, 'FontSize', 11)
text(xMin+0.3, yMax*0.81, sprintf('All: r=%.2f, p=%.3f', rAll, pAllCorr), ...
    'Color', 'k', 'FontSize', 11, 'FontWeight','bold')

% ---------- Estética ----------
grid on
box on
set(gca,'Color',[0.95 0.95 0.95])
xlabel('Wellness Scale', 'FontSize', 12, 'FontWeight', 'bold')
ylabel('Ripple Amplitude (\muV)', 'FontSize', 12, 'FontWeight', 'bold')
title('Correlation: Ripple Amplitude vs Wellness Scale', 'FontSize', 14, 'FontWeight', 'bold')
legend({'CTR','CTR fit','MUT','MUT fit','Global fit'}, 'Location','northeast')
















%% Interpolate power spectrum data to get rid of 50Hz noise for Controls 
% this is only for visualisation purposes - those values will simply be
% used as NaNs for statistical analysis later on


for session_idx = 1:size(control_power_spectrum,2) % size dimension is 2 here because for scalar structures it´s different and 2 is y axis i.e. downward
    session_power = control_power_spectrum(session_idx).pyr_theta_dB;
    datafreq = control_power_spectrum(session_idx).frequencies;
    session_power_int = session_power; % make a new array to append interpolated data as a new variable in the structure later w/o overwriting
    int_idx = datafreq >= 49 & datafreq <= 51; % make a logical array of 0 and 1 where you get the indices where frequency values are between 49 and 51 incl.
    interpoler = griddedInterpolant(datafreq(~int_idx), session_power(~int_idx)); % enter ground truth data (0 indices) i.e 0 to 49 and 51 to 1000 to train interpoler
    session_power_int(int_idx) = interpoler(datafreq(int_idx)); % now save the interpolated values between 49 and 51 Hz into those indices in session_power_int
    % 'interpoler' is the object it is learning on. Now fill in indices
    % from 49 to 51 with interpolated ones by feeding x values to be
    % interpolated (datafreq(int_idx)) and it will give us the int y values
    % i.e. 'session_power_int(int_idx)'. So it is giving us 5 new values
    % which have been plugged in to indixes 49:51 in session_power_int
    control_power_spectrum(session_idx).pyr_theta_dB_interp = session_power_int;
%     plot(control_power_spectrum(session_idx).frequencies,session_power, 'color', 'k', 'LineWidth', 0.1)
%     hold on  
end
%% Interpolate power spectrum data in SLM channel


for session_idx = 1:size(control_power_spectrum,2) % size dimension is 2 here because for scalar structures it´s different and 2 is y axis i.e. downward
    session_power = control_power_spectrum(session_idx).slm_theta_dB;
    datafreq = control_power_spectrum(session_idx).frequencies;
    session_power_int = session_power; % make a new array to append interpolated data as a new variable in the structure later w/o overwriting
    int_idx = datafreq >= 49 & datafreq <= 51; % make a logical array of 0 and 1 where you get the indices where frequency values are between 49 and 51 incl.
    interpoler = griddedInterpolant(datafreq(~int_idx), session_power(~int_idx)); % enter ground truth data (0 indices) i.e 0 to 49 and 51 to 1000 to train interpoler
    session_power_int(int_idx) = interpoler(datafreq(int_idx)); % now save the interpolated values between 49 and 51 Hz into those indices in session_power_int
    % 'interpoler' is the object it is learning on. Now fill in indices
    % from 49 to 51 with interpolated ones by feeding x values to be
    % interpolated (datafreq(int_idx)) and it will give us the int y values
    % i.e. 'session_power_int(int_idx)'. So it is giving us 5 new values
    % which have been plugged in to indixes 49:51 in session_power_int
    control_power_spectrum(session_idx).slm_theta_dB_interp = session_power_int;
%     plot(control_power_spectrum(session_idx).frequencies,session_power, 'color', 'k', 'LineWidth', 0.1)
%     hold on  
end
%% Interpolate power spectrum data to get rid of 50Hz noise for Mutants
% this is only for visualisation purposes - those values will simply be
% used as NaNs for statistical analysis later on


for session_idx = 1:size(mutant_power_spectrum,2) % size dimension is 2 here because for scalar structures it´s different and 2 is y axis i.e. downward
    session_power = mutant_power_spectrum(session_idx).pyr_theta_dB;
    datafreq = mutant_power_spectrum(session_idx).frequencies;
    session_power_int = session_power; % make a new array to append interpolated data as a new variable in the structure later w/o overwriting
    int_idx = datafreq >= 49 & datafreq <= 51; % make a logical array of 0 and 1 where you get the indices where frequency values are between 49 and 51 incl.
    interpoler = griddedInterpolant(datafreq(~int_idx), session_power(~int_idx)); % enter ground truth data (0 indices) i.e 0 to 49 and 51 to 1000 to train interpoler
    session_power_int(int_idx) = interpoler(datafreq(int_idx)); % now save the interpolated values between 49 and 51 Hz into those indices in session_power_int
    % 'interpoler' is the object it is learning on. Now fill in indices
    % from 49 to 51 with interpolated ones by feeding x values to be
    % interpolated (datafreq(int_idx)) and it will give us the int y values
    % i.e. 'session_power_int(int_idx)'. So it is giving us 5 new values
    % which have been plugged in to indixes 49:51 in session_power_int
    mutant_power_spectrum(session_idx).pyr_theta_dB_interp = session_power_int;
%     plot(mutant_power_spectrum(session_idx).frequencies,session_power, 'color', 'k', 'LineWidth', 0.1)
%     hold on  
end
%% Interpolate power spectrum data for mutant SLM layer

for session_idx = 1:size(mutant_power_spectrum,2) % size dimension is 2 here because for scalar structures it´s different and 2 is y axis i.e. downward
    session_power = mutant_power_spectrum(session_idx).slm_theta_dB;
    datafreq = mutant_power_spectrum(session_idx).frequencies;
    session_power_int = session_power; % make a new array to append interpolated data as a new variable in the structure later w/o overwriting
    int_idx = datafreq >= 49 & datafreq <= 51; % make a logical array of 0 and 1 where you get the indices where frequency values are between 49 and 51 incl.
    interpoler = griddedInterpolant(datafreq(~int_idx), session_power(~int_idx)); % enter ground truth data (0 indices) i.e 0 to 49 and 51 to 1000 to train interpoler
    session_power_int(int_idx) = interpoler(datafreq(int_idx)); % now save the interpolated values between 49 and 51 Hz into those indices in session_power_int
    % 'interpoler' is the object it is learning on. Now fill in indices
    % from 49 to 51 with interpolated ones by feeding x values to be
    % interpolated (datafreq(int_idx)) and it will give us the int y values
    % i.e. 'session_power_int(int_idx)'. So it is giving us 5 new values
    % which have been plugged in to indixes 49:51 in session_power_int
    mutant_power_spectrum(session_idx).slm_theta_dB_interp = session_power_int;
%     plot(mutant_power_spectrum(session_idx).frequencies,session_power, 'color', 'k', 'LineWidth', 0.1)
%     hold on  
end

%% Add statistical analysis for AUC during different frequencies e.g. theta, gamma

test_power_ctr = {control_power_spectrum.pyr_theta_dB_interp};
test_power_mut = {mutant_power_spectrum.pyr_theta_dB_interp};

test_power_slm_ctr = {control_power_spectrum.slm_theta_dB_interp};
test_power_slm_mut = {mutant_power_spectrum.slm_theta_dB_interp};

figure
all_ctr = [];
for session_idx = 1:length(test_power_ctr)
    session_power = test_power_ctr{session_idx};
    all_ctr = [all_ctr session_power];
    plot(control_power_spectrum(session_idx).frequencies,session_power, 'color', 'k', 'LineWidth', 0.1)
    hold on  
end

all_mut = [];
for session_idx_mut = 1:length(test_power_mut)
    session_power_mut = test_power_mut{session_idx_mut};
    all_mut = [all_mut session_power_mut];
    plot(mutant_power_spectrum(session_idx_mut).frequencies,session_power_mut, 'color', 'r', 'LineWidth', 0.05)
    hold on
end


figure
all_ctr_slm = [];
for session_idx = 1:length(test_power_slm_ctr)
    session_power = test_power_slm_ctr{session_idx};
    all_ctr_slm = [all_ctr_slm session_power];
    plot(control_power_spectrum(session_idx).frequencies,session_power, 'color', 'k', 'LineWidth', 0.1)
    hold on  
end

all_mut_slm = [];
for session_idx_mut = 1:length(test_power_slm_mut)
    session_power_mut = test_power_slm_mut{session_idx_mut};
    all_mut_slm = [all_mut_slm session_power_mut];
    plot(mutant_power_spectrum(session_idx_mut).frequencies,session_power_mut, 'color', 'b', 'LineWidth', 0.05)
    hold on
end
title('Power Spectrum theta period - SLM')


%% Statistical Analysis
%Replace 50Hz noise with NaNs for statistical analysis i.e. for indices
%between  49 and 51 = NaNs
all_ctr_stats = [];
all_ctr_stats_slm = [];
for session_idx= 1:size(all_ctr,2) 
    datafreq = control_power_spectrum(session_idx).frequencies;
    session_power = control_power_spectrum(session_idx).pyr_theta_dB;
    noise_idx = datafreq >= 49 & datafreq <= 51;
    session_power(noise_idx) = NaN;
    all_ctr_stats = [all_ctr_stats session_power];
end

for session_idx= 1:size(all_ctr_slm,2) 
    datafreq = control_power_spectrum(session_idx).frequencies;
    session_power = control_power_spectrum(session_idx).slm_theta_dB;
    noise_idx = datafreq >= 49 & datafreq <= 51;
    session_power(noise_idx) = NaN;
    all_ctr_stats_slm = [all_ctr_stats_slm session_power];
end

all_mut_stats = [];
all_mut_stats_slm = [];
for session_idx= 1:size(all_mut,2) 
    datafreq = mutant_power_spectrum(session_idx).frequencies;
    session_power = mutant_power_spectrum(session_idx).pyr_theta_dB;
    noise_idx = datafreq >= 49 & datafreq <= 51;
    session_power(noise_idx) = NaN;
    all_mut_stats = [all_mut_stats session_power];
end

for session_idx= 1:size(all_mut_slm,2) 
    datafreq = mutant_power_spectrum(session_idx).frequencies;
    session_power = mutant_power_spectrum(session_idx).slm_theta_dB;
    noise_idx = datafreq >= 49 & datafreq <= 51;
    session_power(noise_idx) = NaN;
    all_mut_stats_slm = [all_mut_stats_slm session_power];
end

% To get area under the curve for low gamma (40 to 60 Hz) and high gamma
% (70 to 90 Hz) and max value
AUC_ctr_t = nan(size(all_ctr,2),1);
AUC_ctr_lg = nan(size(all_ctr,2),1);
AUC_ctr_hg = nan(size(all_ctr,2),1);
AUC_ctr_extg = nan(size(all_ctr,2),1);
AUC_ctr_th = nan(size(all_ctr,2),1);
AUC_ctr_thn = nan(size(all_ctr,2),1);
max_ctr_th = nan(size(all_ctr,2),1);
max_ctr_thn = nan(size(all_ctr,2),1);
max_ctr_lg = nan(size(all_ctr,2),1);
max_ctr_hg = nan(size(all_ctr,2),1);
max_ctr_extg = nan(size(all_ctr,2),1);
idx_max_ctr_th = nan(size(all_ctr,2),1);
idx_max_ctr_extg = nan(size(all_ctr,2),1);
thetagamma_ratio_ctr = nan(size(all_ctr,2),1);

% Same for SLM layer - initialise lists
AUC_ctr_t_slm = nan(size(all_ctr_slm,2),1);
AUC_ctr_lg_slm = nan(size(all_ctr_slm,2),1);
AUC_ctr_hg_slm = nan(size(all_ctr_slm,2),1);
AUC_ctr_extg_slm = nan(size(all_ctr_slm,2),1);
AUC_ctr_th_slm = nan(size(all_ctr_slm,2),1);
AUC_ctr_thn_slm = nan(size(all_ctr_slm,2),1);
max_ctr_th_slm = nan(size(all_ctr_slm,2),1);
max_ctr_thn_slm = nan(size(all_ctr_slm,2),1);
max_ctr_lg_slm = nan(size(all_ctr_slm,2),1);
max_ctr_hg_slm = nan(size(all_ctr_slm,2),1);
max_ctr_extg_slm = nan(size(all_ctr_slm,2),1);
idx_max_ctr_th = nan(size(all_ctr_slm,2),1);
idx_max_ctr_extg = nan(size(all_ctr_slm,2),1);
idx_max_ctr_th_slm = nan(size(all_ctr_slm,2),1);
idx_max_ctr_extg_slm = nan(size(all_ctr_slm,2),1);
thetagamma_ratio_ctr_slm = nan(size(all_ctr_slm,2),1);


for session_idx= 1:size(all_ctr_stats,2) %for each session in control list
    x = control_power_spectrum(session_idx).frequencies; %frequency indexes
    y = all_ctr_stats(:,session_idx);
    y = nanmean(y,2);
    plot(x,y);
    xlim([0 100])
    AUC_ctr_t(session_idx) = trapz(x,y); %AUC_ctr_t = total AUC for entire power spectrum curve
    % To get AUC for certain portions e.g. 40-60Hz or 70-90Hz
    
    idx_4 = find(x>=4);
    idx_4 = idx_4(1);
    
    idx_6 = find(x>=6);
    idx_6 = idx_6(1);
    
    idx_10 = find(x>=10);
    idx_10 = idx_10(1);
    
    idx_12 = find(x>=12);
    idx_12 = idx_12(1);
    
    idx_40 = find(x>=40);
    idx_40 = idx_40(1);

    idx_60 = find(x>=60);
    idx_60 = idx_60(1);
    
    idx_70 = find(x>=70);
    idx_70 = idx_70(1);
    
    idx_80 = find(x>=80);
    idx_80 = idx_80(1);

    idx_90 = find(x>=90);
    idx_90 = idx_90(1);
    
    
    AUC_ctr_th(session_idx) = trapz(x(idx_4:idx_12), y(idx_4:idx_12)); %extended theta
    AUC_ctr_thn(session_idx) = trapz(x(idx_6:idx_10), y(idx_6:idx_10)); %narrow theta
    idnonan = idx_40:idx_60; idnonan(isnan(y(idnonan))) = [];
    idnonan_extg = idx_40:idx_80; idnonan_extg(isnan(y(idnonan_extg))) = [];
    AUC_ctr_lg(session_idx) = trapz(x(idnonan), y(idnonan)); %slow/low gamma 40-60 Hz
    AUC_ctr_extg(session_idx) = trapz(x(idnonan_extg), y(idnonan_extg)); %extended gamma 40-80 Hz
    AUC_ctr_hg(session_idx) = trapz(x(idx_70:idx_90), y(idx_70:idx_90)); %high/fast gamma 60-90 Hz
    [max_ctr_th(session_idx), idx_max_ctr_th(session_idx)] = max(y(idx_6:idx_10));
    max_ctr_thn(session_idx) = max(y(idx_4:idx_12));
    max_ctr_lg(session_idx)= max(y(idx_40:idx_60));
    max_ctr_hg(session_idx) = max(y(idx_70:idx_90));
    max_ctr_extg(session_idx) = max(y(idx_40:idx_80));
    [max_ctr_extg(session_idx), idx_max_ctr_extg(session_idx)] = max(y(idx_40:idx_80));
    thetagamma_ratio_ctr(session_idx) = max_ctr_th(session_idx)/max_ctr_extg(session_idx);
    
    control_power_spectrum(session_idx).AUC_ctr_th = AUC_ctr_th(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_thn = AUC_ctr_thn(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_lg = AUC_ctr_lg(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_extg = AUC_ctr_extg(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_hg = AUC_ctr_hg(session_idx);
   
    control_power_spectrum(session_idx).max_ctr_th = max_ctr_th(session_idx);
    control_power_spectrum(session_idx).idx_max_ctr_th = x(idx_6-1+idx_max_ctr_th(session_idx));
    control_power_spectrum(session_idx).max_ctr_extg = max_ctr_extg(session_idx);
    control_power_spectrum(session_idx).idx_max_ctr_extg = x(idx_40-1+idx_max_ctr_extg(session_idx));
    
    control_power_spectrum(session_idx).max_ctr_thn = max_ctr_thn(session_idx);
    control_power_spectrum(session_idx).max_ctr_lg = max_ctr_lg(session_idx);
    control_power_spectrum(session_idx).max_ctr_hg = max_ctr_hg(session_idx);
    control_power_spectrum(session_idx).thetagamma_ratio_ctr = thetagamma_ratio_ctr(session_idx);
    
end



for session_idx= 1:size(all_ctr_stats_slm,2) %for each session in control list
    x = control_power_spectrum(session_idx).frequencies; %frequency indexes
    y = all_ctr_stats_slm(:,session_idx);
    y = nanmean(y,2);
    plot(x,y);
    xlim([0 100])
    AUC_ctr_t_slm(session_idx) = trapz(x, y); %AUC_ctr_t = total AUC for entire power spectrum curve
    % To get AUC for certain portions e.g. 40-60Hz or 70-90Hz
    
    idx_4 = find(x>=4);
    idx_4 = idx_4(1);
    
    idx_6 = find(x>=6);
    idx_6 = idx_6(1);
    
    idx_10 = find(x>=10);
    idx_10 = idx_10(1);
    
    idx_12 = find(x>=12);
    idx_12 = idx_12(1);
    
    idx_40 = find(x>=40);
    idx_40 = idx_40(1);

    idx_60 = find(x>=60);
    idx_60 = idx_60(1);
    
    idx_70 = find(x>=70);
    idx_70 = idx_70(1);
    
    idx_80 = find(x>=80);
    idx_80 = idx_80(1);

    idx_90 = find(x>=90);
    idx_90 = idx_90(1);
    
    
    AUC_ctr_th_slm(session_idx) = trapz(x(idx_4:idx_12), y(idx_4:idx_12)); %extended theta
    AUC_ctr_thn_slm(session_idx) = trapz(x(idx_6:idx_10), y(idx_6:idx_10)); %narrow theta
    idnonan = idx_40:idx_60; idnonan(isnan(y(idnonan))) = [];
    idnonan_extg = idx_40:idx_80; idnonan_extg(isnan(y(idnonan_extg))) = [];
    AUC_ctr_lg_slm(session_idx) = trapz(x(idnonan), y(idnonan)); %slow/low gamma 40-60 Hz
    AUC_ctr_extg_slm(session_idx) = trapz(x(idnonan_extg), y(idnonan_extg)); %extended gamma 40-80 Hz
    AUC_ctr_hg_slm(session_idx) = trapz(x(idx_70:idx_90), y(idx_70:idx_90)); %high/fast gamma 60-90 Hz
   
    max_ctr_thn_slm(session_idx) = max(y(idx_4:idx_12));
    max_ctr_lg_slm(session_idx) = max(y(idx_40:idx_60));
    max_ctr_hg_slm(session_idx) = max(y(idx_70:idx_90));
    [max_ctr_th_slm(session_idx), idx_max_ctr_th_slm(session_idx)] = max(y(idx_6:idx_10));
    [max_ctr_extg_slm(session_idx), idx_max_ctr_extg_slm(session_idx)] = max(y(idx_40:idx_80));
    thetagamma_ratio_ctr_slm(session_idx) = max_ctr_th_slm(session_idx)/max_ctr_extg_slm(session_idx);
    
    
    control_power_spectrum(session_idx).AUC_ctr_th_slm = AUC_ctr_th_slm(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_thn_slm = AUC_ctr_thn_slm(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_lg_slm = AUC_ctr_lg_slm(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_extg_slm = AUC_ctr_extg_slm(session_idx);
    control_power_spectrum(session_idx).AUC_ctr_hg_slm = AUC_ctr_hg_slm(session_idx);
    control_power_spectrum(session_idx).max_ctr_th_slm = max_ctr_th_slm(session_idx);
    control_power_spectrum(session_idx).max_ctr_thn_slm = max_ctr_thn_slm(session_idx);
    control_power_spectrum(session_idx).max_ctr_lg_slm = max_ctr_lg_slm(session_idx);
    control_power_spectrum(session_idx).max_ctr_hg_slm = max_ctr_hg_slm(session_idx);
    control_power_spectrum(session_idx).max_ctr_extg_slm = max_ctr_extg_slm(session_idx);
    
    control_power_spectrum(session_idx).max_ctr_th_slm = max_ctr_th_slm(session_idx);
    control_power_spectrum(session_idx).idx_max_ctr_th_slm = x(idx_6-1+idx_max_ctr_th_slm(session_idx));
    control_power_spectrum(session_idx).max_ctr_extg_slm = max_ctr_extg_slm(session_idx);
    control_power_spectrum(session_idx).idx_max_ctr_extg_slm = x(idx_40-1+idx_max_ctr_extg_slm(session_idx));
    control_power_spectrum(session_idx).thetagamma_ratio_ctr_slm = thetagamma_ratio_ctr_slm(session_idx);
end



%% Same again for mutant sessions - get measures in diff bandwidths for pyr and slm
AUC_mut_t = nan(size(all_mut,2),1); % make matrix of size of mut sessions by 1 column ((all_mut,2),1)
AUC_mut_lg = nan(size(all_mut,2),1);
AUC_mut_hg = nan(size(all_mut,2),1);
AUC_mut_extg = nan(size(all_mut,2),1);
AUC_mut_th = nan(size(all_mut,2),1);
AUC_mut_thn = nan(size(all_mut,2),1);
max_mut_th = nan(size(all_mut,2),1);
max_mut_thn = nan(size(all_mut,2),1);
max_mut_lg = nan(size(all_mut,2),1);
max_mut_hg = nan(size(all_mut,2),1);
max_mut_extg = nan(size(all_mut,2),1);
thetagamma_ratio_mut = nan(size(all_mut,2),1);

AUC_mut_t_slm = nan(size(all_mut_slm,2),1); % make matrix of size of mut sessions by 1 column ((all_mut,2),1)
AUC_mut_lg_slm = nan(size(all_mut_slm,2),1);
AUC_mut_hg_slm = nan(size(all_mut_slm,2),1);
AUC_mut_extg_slm = nan(size(all_mut_slm,2),1);
AUC_mut_th_slm = nan(size(all_mut_slm,2),1);
AUC_mut_thn_slm = nan(size(all_mut_slm,2),1);
max_mut_th_slm = nan(size(all_mut_slm,2),1);
max_mut_thn_slm = nan(size(all_mut_slm,2),1);
max_mut_lg_slm = nan(size(all_mut_slm,2),1);
max_mut_hg_slm = nan(size(all_mut_slm,2),1);
max_mut_extg_slm = nan(size(all_mut_slm,2),1);
thetagamma_ratio_mut_slm = nan(size(all_mut_slm,2),1);

idx_max_mut_th = nan(size(all_mut,2),1);
idx_max_mut_extg = nan(size(all_mut,2),1);
idx_max_mut_th_slm = nan(size(all_mut_slm,2),1);
idx_max_mut_extg_slm = nan(size(all_mut_slm,2),1);


for session_idx= 1:size(all_mut_stats,2) %for each session in mutant list
    x = mutant_power_spectrum(session_idx).frequencies;
    y = all_mut_stats(:,session_idx);
    y = nanmean(y,2);
    plot(x,y);
    xlim([0 100])
    AUC_mut_t(session_idx) = trapz(x, y);
    % To get AUC for certain portions e.g. 40-60Hz or 70-90Hz and also
    % extended theta 4-12 Hz and narrower theta 6-10Hz
    % Find index within 
    idx_4 = find(x>=4);
    idx_4 = idx_4(1);
    
    idx_6 = find(x>=6);
    idx_6 = idx_6(1);
    
    idx_10 = find(x>=10);
    idx_10 = idx_10(1);
    
    idx_12 = find(x>=12);
    idx_12 = idx_12(1);
    
    idx_40 = find(x>=40);
    idx_40 = idx_40(1);

    idx_60 = find(x>=60);
    idx_60 = idx_60(1);
    
    idx_70 = find(x>=70);
    idx_70 = idx_70(1);
    
    idx_80 = find(x>=80);
    idx_80 = idx_80(1);

    idx_90 = find(x>=90);
    idx_90 = idx_90(1);
    
    AUC_mut_th(session_idx) = trapz(x(idx_4:idx_12), y(idx_4:idx_12)); %extended theta
    AUC_mut_thn(session_idx) = trapz(x(idx_6:idx_10), y(idx_6:idx_10)); %narrow theta
    idnonan = idx_40:idx_60; idnonan(isnan(y(idnonan))) = [];
    idnonan_extg = idx_40:idx_80; idnonan_extg(isnan(y(idnonan_extg))) = [];
    AUC_mut_lg(session_idx) = trapz(x(idnonan), y(idnonan)); %slow/low gamma 40-60 Hz
    AUC_mut_extg(session_idx) = trapz(x(idnonan_extg), y(idnonan_extg)); %extended gamma 40-80 Hz
    AUC_mut_hg(session_idx) = trapz(x(idx_70:idx_90), y(idx_70:idx_90));
    
    max_mut_thn(session_idx) = max(y(idx_4:idx_12));
    max_mut_lg(session_idx) = max(y(idx_40:idx_60));
    max_mut_hg(session_idx) = max(y(idx_70:idx_90));
    
    
    [max_mut_th(session_idx), idx_max_mut_th(session_idx)] = max(y(idx_6:idx_10));
    [max_mut_extg(session_idx), idx_max_mut_extg(session_idx)] = max(y(idx_40:idx_80));
    thetagamma_ratio_mut(session_idx) = max_mut_th(session_idx)/max_mut_extg(session_idx);
    
    mutant_power_spectrum(session_idx).AUC_mut_th = AUC_mut_th(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_thn = AUC_mut_thn(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_lg= AUC_mut_lg(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_extg = AUC_mut_extg(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_hg = AUC_mut_hg(session_idx);
    mutant_power_spectrum(session_idx).max_mut_th = max_mut_th(session_idx);
    mutant_power_spectrum(session_idx).max_mut_thn = max_mut_thn(session_idx);
    mutant_power_spectrum(session_idx).max_mut_lg = max_mut_lg(session_idx);
    mutant_power_spectrum(session_idx).max_mut_hg = max_mut_hg(session_idx);
    mutant_power_spectrum(session_idx).max_mut_extg = max_mut_extg(session_idx);
    
    mutant_power_spectrum(session_idx).max_mut_th = max_mut_th(session_idx);
    mutant_power_spectrum(session_idx).idx_max_mut_th = x(idx_6-1+idx_max_mut_th(session_idx));
    mutant_power_spectrum(session_idx).max_mut_extg = max_mut_extg(session_idx);
    mutant_power_spectrum(session_idx).idx_max_mut_extg = x(idx_40-1+idx_max_mut_extg(session_idx));
    mutant_power_spectrum(session_idx).thetagamma_ratio_mut = thetagamma_ratio_mut(session_idx);
end



for session_idx= 1:size(all_mut_stats_slm,2) %for each session in mutant list
    x = mutant_power_spectrum(session_idx).frequencies;
    y = all_mut_stats_slm(:,session_idx);
    y = nanmean(y,2);
    plot(x,y);
    xlim([0 100])
    AUC_mut_t_slm(session_idx) = trapz(x, y);
    % To get AUC for certain portions e.g. 40-60Hz or 70-90Hz and also
    % extended theta 4-12 Hz and narrower theta 6-10Hz
    % Find index within 
    idx_4 = find(x>=4);
    idx_4 = idx_4(1);
    
    idx_6 = find(x>=6);
    idx_6 = idx_6(1);
    
    idx_10 = find(x>=10);
    idx_10 = idx_10(1);
    
    idx_12 = find(x>=12);
    idx_12 = idx_12(1);
    
    idx_40 = find(x>=40);
    idx_40 = idx_40(1);

    idx_60 = find(x>=60);
    idx_60 = idx_60(1);
    
    idx_70 = find(x>=70);
    idx_70 = idx_70(1);
    
    idx_80 = find(x>=80);
    idx_80 = idx_80(1);

    idx_90 = find(x>=90);
    idx_90 = idx_90(1);
    
    AUC_mut_th_slm(session_idx) = trapz(x(idx_4:idx_12), y(idx_4:idx_12)); %extended theta
    AUC_mut_thn_slm(session_idx) = trapz(x(idx_6:idx_10), y(idx_6:idx_10)); %narrow theta
    idnonan = idx_40:idx_60; idnonan(isnan(y(idnonan))) = [];
    idnonan_extg = idx_40:idx_80; idnonan_extg(isnan(y(idnonan_extg))) = [];
    AUC_mut_lg_slm(session_idx) = trapz(x(idnonan), y(idnonan)); %slow/low gamma 40-60 Hz
    AUC_mut_extg_slm(session_idx) = trapz(x(idnonan_extg), y(idnonan_extg)); %extended gamma 40-80 Hz
    AUC_mut_hg_slm(session_idx) = trapz(x(idx_70:idx_90), y(idx_70:idx_90));
   
    max_mut_thn_slm(session_idx) = max(y(idx_4:idx_12));
    max_mut_lg_slm(session_idx) = max(y(idx_40:idx_60));
    max_mut_hg_slm(session_idx) = max(y(idx_70:idx_90));
   
    
    [max_mut_th_slm(session_idx), idx_max_mut_th_slm(session_idx)] = max(y(idx_6:idx_10));
    [max_mut_extg_slm(session_idx), idx_max_mut_extg_slm(session_idx)] = max(y(idx_40:idx_80));
    thetagamma_ratio_mut_slm(session_idx) = max_mut_th_slm(session_idx)/max_mut_extg_slm(session_idx);
    
    mutant_power_spectrum(session_idx).AUC_mut_th_slm = AUC_mut_th_slm(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_thn_slm = AUC_mut_thn_slm(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_lg_slm = AUC_mut_lg_slm(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_extg_slm = AUC_mut_extg_slm(session_idx);
    mutant_power_spectrum(session_idx).AUC_mut_hg_slm = AUC_mut_hg_slm(session_idx);
    mutant_power_spectrum(session_idx).max_mut_th_slm = max_mut_th_slm(session_idx);
    mutant_power_spectrum(session_idx).max_mut_thn_slm = max_mut_thn_slm(session_idx);
    mutant_power_spectrum(session_idx).max_mut_lg_slm = max_mut_lg_slm(session_idx);
    mutant_power_spectrum(session_idx).max_mut_hg_slm = max_mut_hg_slm(session_idx);
    mutant_power_spectrum(session_idx).max_mut_extg_slm = max_mut_extg_slm(session_idx);
    
    mutant_power_spectrum(session_idx).max_mut_th_slm = max_mut_th_slm(session_idx);
    mutant_power_spectrum(session_idx).idx_max_mut_th_slm = x(idx_6-1+idx_max_mut_th_slm(session_idx));
    mutant_power_spectrum(session_idx).max_mut_extg_slm = max_mut_extg_slm(session_idx);
    mutant_power_spectrum(session_idx).idx_max_mut_extg_slm = x(idx_40-1+idx_max_mut_extg_slm(session_idx));
    mutant_power_spectrum(session_idx).thetagamma_ratio_mut_slm = thetagamma_ratio_mut_slm(session_idx);
end








%%
clearvars -except working_dir control_power_spectrum mutant_power_spectrum
save(fullfile(working_dir,'control_power_spectrum.mat'),'control_power_spectrum');
save(fullfile(working_dir,'mutant_power_spectrum.mat'), 'mutant_power_spectrum');


%% Plot correlations of speed to theta power:
dirAnalyset = 'G:\Users\Laboratorio\Documents\GitHub\Analyset';
addpath(genpath(dirAnalyset))

figure

scatter([mutant_power_spectrum.mean_speed],[mutant_power_spectrum.max_mut_th], 'o')
hold on 
groupCorr([mutant_power_spectrum.mean_speed],[mutant_power_spectrum.max_mut_th])
hold on 
groupCorr([control_power_spectrum.mean_speed],[control_power_spectrum.max_ctr_th])
xlabel('Mean speed (cm/s)')
ylabel('Max theta power (dB)')


figure
groupCorr([mutant_power_spectrum.mean_speed],[mutant_power_spectrum.AUC_mut_th],'inAxis',true, 'MarkerEdgeColor',[0/255 66/255 168/255], 'MarkerSize', 15, 'MarkerColor', 'none')
hold on
groupCorr([control_power_spectrum.mean_speed],[control_power_spectrum.AUC_ctr_th],'MarkerEdgeColor','k', 'MarkerColor','none',...
    'inAxis', true,  'MarkerSize', 15);

xlabel('Mean speed (cm/s)')
ylabel('Max theta power (dB)')
legend('Mutant','','Control','')
