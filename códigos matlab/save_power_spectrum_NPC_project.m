%% 1. INITIAL SETTINGS & CONFIGURATION
clear; close all; clc;

% Set your working directory
working_dir = 'D:\Violeta\ANALISIS';
cd(working_dir);

% Create results folder if it doesn't exist
if ~isfolder('RESULTS'), mkdir('RESULTS'); end

% Define analysis parameters
conditions = {'WT', 'NPC', 'EFV'};
group_colors = [76 153 102; 70 120 180; 150 110 170] / 256; % WT, NPC, EFV
psFreqs = 1:0.5:1000;

% Frequency Bands Definition
bands = struct(...
    'Delta', [0 3], ...
    'Theta', [4 12], ...
    'Alpha', [9 16], ...
    'SlowGamma', [20 40], ...
    'MidGamma', [40 60], ...
    'FastGamma', [60 80], ...
    'Ripple', [150 250]);

%% 2. SESSION CLASSIFICATION
files = dir(working_dir);
files = files([files.isdir] & ~ismember({files.name}, {'.', '..'}));

sessions = struct([]);
for i = 1:length(files)
    animalName = files(i).name;
    % Identify group/condition
    idxCond = find(cellfun(@(c) contains(animalName, c, 'IgnoreCase', true), conditions));
    
    if ~isempty(idxCond)
        sessFiles = dir(fullfile(files(i).folder, animalName));
        sessFiles([sessFiles.isdir] == 0 | ismember({sessFiles.name}, {'.', '..'})) = [];
        
        for j = 1:length(sessFiles)
            index = numel(sessions) + 1;
            sessions(index).session = fullfile(sessFiles(j).folder, sessFiles(j).name);
            sessions(index).animal = animalName;
            sessions(index).condition = conditions{idxCond};
        end
    end
end

%% 3. UNIFIED DATA PROCESSING
analysisResults = struct(); 

for iCond = 1:length(conditions)
    currentCond = conditions{iCond};
    condSess = sessions(strcmp({sessions.condition}, currentCond));
    tempAnalysis = struct([]);
    
    fprintf('Processing condition: %s...\n', currentCond);
    
    for iSess = 1:length(condSess)
        sessPath = condSess(iSess).session;
        analysetFolder = fullfile(sessPath, 'analyset');
        
        if isfolder(analysetFolder)
            % Dynamic loading of required .mat files
            matFiles = dir(fullfile(analysetFolder, '*.mat'));
            targets = {'ripple', 'configChannels', 'theta', 'psProfile'};
            for f = 1:length(matFiles)
                if any(cellfun(@(t) contains(matFiles(f).name, t), targets))
                    load(fullfile(matFiles(f).folder, matFiles(f).name));
                end
            end
            
            % Window calculation (Vectorized logic)
            fs = 2500;
            wSize = psProfile.params.tWin/fs;
            wStep = psProfile.params.dtWin/fs;
            pSbins = [(0:wStep:(size(psProfile.pS,2)-1)*wStep)', (0:wStep:(size(psProfile.pS,2)-1)*wStep)' + wSize];
            
            % Helper function to find power spectrum windows within specific time events
            calcWindows = @(times) any((times(:,1) < pSbins(:,2)') & (times(:,2) > pSbins(:,1)'), 1)';
            
            pSwinTheta = calcWindows(thetaLIAAnalysis.timesTheta);
            pSwinLIA   = calcWindows(thetaLIAAnalysis.timesLIA);
            
            % Save processed data into temporary structure
            tempAnalysis(iSess).pSTheta = squeeze(mean(psProfile.pS(:, logical(pSwinTheta), :), 2));
            tempAnalysis(iSess).pSLIA   = squeeze(mean(psProfile.pS(:, logical(pSwinLIA), :), 2));
            tempAnalysis(iSess).ripples = rippleAnalysis.propRipS;
            tempAnalysis(iSess).channelInfo = configChannels.ch;
            tempAnalysis(iSess).animal = condSess(iSess).animal;
        end
    end
    analysisResults.(currentCond) = tempAnalysis;
end

%% 4. GENERAL POWER SPECTRUM PROFILE (0-300 Hz)
figPS = figure('Name', 'Power Spectrum Profile', 'Position', [100 100 1000 800]);
t = tiledlayout(2,2);
title(t, 'Power Spectrum Group Comparison (Mean ± SEM)');

layers = {'pyr', 'theta'}; % 'theta' is SLM in your config
states = {'pSTheta', 'pSLIA'};
layer_names = {'PYR', 'SLM'};
state_names = {'Theta', 'LIA'};

for iEst = 1:2
    for iCap = 1:2
        nexttile; hold on;
        for g = 1:3
            res = analysisResults.(conditions{g});
            currentData = [];
            for s = 1:length(res)
                if isfield(res(s), 'channelInfo') && ~isempty(res(s).channelInfo)
                    ch = res(s).channelInfo.(layers{iCap});
                    currentData = [currentData, res(s).(states{iEst})(:, ch)];
                end
            end
            
            if ~isempty(currentData)
                line_style = '-'; if iEst == 2, line_style = '--'; end
                plotFill(psFreqs, currentData, 'color', group_colors(g,:), 'LineStyle', line_style);
            end
        end
        xlim([0 80]); yscale log; grid on;
        xlabel('Frequency (Hz)'); ylabel('Power (dB)');
        title(sprintf('%s Channel - %s State', layer_names{iCap}, state_names{iEst}));
        if iEst == 1 && iCap == 1, legend({'', 'WT', '', 'NPC', '', 'EFV'}); end
    end
end
saveas(figPS, fullfile('RESULTS', 'PowerSpectrum_Overview.png'));

%% 5. AUTOMATED BAND STATISTICS (Mean & Max)
band_names = fieldnames(bands);
% Get absolute path to the results folder
results_path = fullfile(working_dir, 'RESULTS');
if ~isfolder(results_path), mkdir(results_path); end

for iB = 1:length(band_names)
    bName = band_names{iB};
    idxF = psFreqs >= bands.(bName)(1) & psFreqs <= bands.(bName)(2);
    
    for iEst = 1:2
        for iCap = 1:2
            dataMean = cell(1,3); dataMax = cell(1,3);
            for g = 1:3
                res = analysisResults.(conditions{g});
                m_v = []; mx_v = [];
                for s = 1:length(res)
                    if isfield(res(s), 'channelInfo') && ~isempty(res(s).channelInfo)
                        ch = res(s).channelInfo.(layers{iCap});
                        segment = res(s).(states{iEst})(idxF, ch);
                        % Handle empty segments to avoid mean([]) = NaN
                        if ~isempty(segment)
                            m_v = [m_v; mean(segment)];
                            mx_v = [mx_v; max(segment)];
                        end
                    end
                end
                dataMean{g} = m_v; dataMax{g} = mx_v;
            end
            
            % Generate Figure
            figTitle = sprintf('%s Power %s %s', bName, layer_names{iCap}, state_names{iEst});
            fig = figure('Position', [123, 65, 305, 570], 'Visible', 'off');
            t = tiledlayout(2,1);
            
            % Subplot 1: MEAN
            nexttile;
            groupStats(dataMean, [], 'plotType', 'boxplot', 'color', group_colors, 'inAxis', true, 'sigStar', true);
            hold on;
            for g = 1:3
                if ~isempty(dataMean{g})
                    scatter(ones(size(dataMean{g}))*g, dataMean{g}, 'k', 'jitter', 'on'); 
                end
            end
            title('Mean Power'); ylabel('dB'); xticklabels(conditions);
            
            % Subplot 2: MAX
            nexttile;
            groupStats(dataMax, [], 'plotType', 'boxplot', 'color', group_colors, 'inAxis', true, 'sigStar', true);
            hold on;
            for g = 1:3
                if ~isempty(dataMax{g})
                    scatter(ones(size(dataMax{g}))*g, dataMax{g}, 'k', 'jitter', 'on'); 
                end
            end
            title('Peak Power (Max)'); ylabel('dB'); xticklabels(conditions);
            
            sgtitle(figTitle);
            
            % CLEAN FILENAME: Remove any colons or slashes
            saveName = strrep(figTitle, ':', '');
            saveName = strrep(saveName, ' ', '_');
            
            % USE FULL ABSOLUTE PATH
            full_save_path = fullfile(results_path, [saveName '.png']);
            
            % SAVE
            saveas(fig, full_save_path);
            close(fig);
            
            fprintf('Saved: %s\n', saveName);
        end
    end
end

%% 6. RIPPLE FEATURES ANALYSIS (Multi-panel: Frequency, Amplitude, Power)
condNames = {'WT', 'NPC', 'EFV'};
featureNames = {'Frequency (Hz)', 'Amplitude (uV)', 'Power (dB)'};
dataRipples = cell(3, 3); % Rows: Features (Freq, Amp, Pow) | Cols: Groups (WT, NPC, EFV)

for g = 1:3
    res = analysisResults.(condNames{g});
    feat1 = []; feat2 = []; feat3 = [];
    
    for s = 1:length(res)
        if isfield(res(s), 'ripples') && ~isempty(res(s).ripples)
            % Extract data and convert to matrix if it's a table
            current_rip = res(s).ripples;
            if istable(current_rip) || istimetable(current_rip), current_rip = table2array(current_rip); end
            
            % Calculate session means for each column (assuming 3 columns)
            % Column 1: Freq, Column 2: Amp, Column 3: Power
            feat1 = [feat1; mean(current_rip(:, 1), 'omitnan')];
            feat2 = [feat2; mean(current_rip(:, 2), 'omitnan')];
            feat3 = [feat3; mean(current_rip(:, 3), 'omitnan')];
        end
    end
    dataRipples{1, g} = feat1;
    dataRipples{2, g} = feat2;
    dataRipples{3, g} = feat3;
end

% --- Generate Figure (Matching your uploaded image) ---
if ~all(cellfun(@isempty, dataRipples(:)))
    figRip = figure('Name', 'Ripple Features', 'Position', [100, 50, 450, 900]);
    t = tiledlayout(3, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, 'Ripple Features', 'FontSize', 16);

    for iFeat = 1:3
        nexttile;
        % Get data for the 3 groups for this specific feature
        plotData = dataRipples(iFeat, :);
        
        % Check if we have data to plot
        if ~all(cellfun(@isempty, plotData))
            % Call your groupStats function
            groupStats(plotData, [], 'plotType', 'boxplot', 'color', group_colors, ...
                       'inAxis', true, 'sigStar', true);
            hold on;
            
            % Overlay individual session points (Scatter)
            for g = 1:3
                if ~isempty(plotData{g})
                    y_val = plotData{g};
                    x_pos = ones(size(y_val)) * g;
                    % Add jitter to match the style
                    x_jitter = x_pos + (0.1 * randn(size(x_pos)));
                    scatter(x_jitter, y_val, 40, 'k', 'LineWidth', 0.8);
                end
            end
            
            ylabel(featureNames{iFeat});
            xticklabels(condNames);
            grid on;
            set(gca, 'Box', 'off'); % Cleaner look
        end
    end
    
    % Save with absolute path
    full_save_path_rip = fullfile(working_dir, 'RESULTS', 'Ripple_Features_Detailed.png');
    saveas(figRip, full_save_path_rip);
    fprintf('Saved: Ripple_Features_Detailed.png\n');
end