
function processRippleSession(sessionPath, so_ch, pyr_ch, rad_ch, slm_ch);


analysetPath = fullfile(sessionPath, 'analyset');
load(fullfile(analysetPath, 'rippleAnalysis.mat'));
windowSize = 0.05;
spontMarginSizes = [0.02, 0.02];
evokedMarginSizes = [-0.025, -0.025];
evokedMiddle = 0.035;
slack = 0.008; 
load(fullfile(analysetPath,'basicData.mat'));
fs = basicData.fsLFP;
windowHalfSize = ceil(windowSize/2 * fs);
spontMarginSizes = round(spontMarginSizes * fs);
%evokedMarginSizes = round(evokedMarginSizes * fs);
evokedMiddle = round(evokedMiddle * fs);
slack = round(slack * fs);


% filter signal and compute envelope
load(fullfile(analysetPath,'configChannels.mat'));
numShanks = length(configChannels.ch.ripple);
filtLFP = cell(1, numShanks);
ripEnv = cell(1, numShanks);

N = 20;  % order
fStop1 = 65;  % first stopband 	
fPass1 = 70;  % first passband frequency
fPass2 = 400;  % second passband frequency
fStop2 = 422;  % second stopband frequency
wStop1 = 1;  % first stopband weight
wPass = 1;  % passband weight
wStop2 = 1;  % second stopband weight

bRP = firls(N, [0 fStop1 fPass1 fPass2 fStop2 fs/2]/(fs/2), [0 0 1 1 0 0], [wStop1 wPass wStop2]);

frameLength = 2*floor(0.0334*fs/2)+1;  % savitzky-golay filtering frame length, rounded to nearest odd number

datFile = 'LFP_downsampled.dat';
numChannels = max(configChannels.chMap);


if configChannels.probeType == 8

    % ---- Buscar carpeta que empiece por "Record"
    recordDir = dir(fullfile(sessionPath, 'Record*'));
    recordDir = recordDir([recordDir.isdir]);

    if isempty(recordDir)
        error('No Record* folder found in %s', sessionPath);
    end

    recordPath = fullfile(sessionPath, recordDir(1).name);

    % ---- Ir a continuous
    contPath = fullfile(recordPath, ...
        'experiment1', 'recording1', 'continuous');

    if ~exist(contPath, 'dir')
        error('continuous folder not found in %s', contPath);
    end

    % ---- Buscar carpeta que empiece por Intan o Acquisition
    dataDir = dir(fullfile(contPath, 'Intan*'));
    if isempty(dataDir)
        dataDir = dir(fullfile(contPath, 'Acquisition*'));
    end

    if isempty(dataDir)
        error('No Intan* or Acquisition* folder found in %s', contPath);
    end

    datPath = fullfile(contPath, dataDir(1).name);

elseif configChannels.probeType == 1

    % El .dat está en analyset
    datPath = sessionPath;

else
    error('Unknown probeType')
end

% ---- Ruta completa al archivo
fullDatFile = fullfile(datPath, datFile);

if ~exist(fullDatFile, 'file')
    error('%s not found', fullDatFile);
end

% ---- Cargar datos
dataLFP = bz_LoadBinary(fullDatFile, 'nChannels', numChannels);
dataLFP = double(dataLFP);

for shankNum = unique(rippleAnalysis.shRipS)'%%%%%%%%%%%%%%%%%%unique(ripplesTable.shank)'
    filtLFP{shankNum} = filtfilt(bRP, 1, dataLFP(:, configChannels.ch.ripple(shankNum)));

    envelope = filtLFP{shankNum} .* filtLFP{shankNum};
    envelope = movmean(movmean(sgolayfilt(envelope, 4, frameLength), 0.0030*fs), 0.0065*fs);
    ripEnv{shankNum} = envelope;
end


% re-align ripples 

numRipples = height(rippleAnalysis.iRipS);
rawRipples = nan(numRipples, 2*windowHalfSize+1);
filtRipples = nan(numRipples, 2*windowHalfSize+1);
alignedRippleTimes = nan(numRipples, 3);
RipStart = rippleAnalysis.iRipS(:,1);
RipEnd = rippleAnalysis.iRipS(:,2);

for rippleNum = 1:numRipples

    shankNum = rippleAnalysis.shRipS(rippleNum);
    marginSizes = spontMarginSizes;
    rippleIdx = (RipStart(rippleNum)-marginSizes(1)):(RipEnd(rippleNum)+marginSizes(2));
    env = ripEnv{shankNum}(rippleIdx);
    [~, iMax] = max(env);
    iMiddle = rippleIdx(1) + iMax - 1;

   
    % align to a minimum

    % [valleys, valleyLocs] = findpeaks(-basicData.dataLFP((iMiddle - slack):(iMiddle + slack), configChannels.ch.ripple(shankNum)));
    [valleys, valleyLocs] = findpeaks(-filtLFP{shankNum}((iMiddle - slack):(iMiddle + slack)));
    [~, iLowest] = max(valleys);
    iMinimum = iMiddle - slack + valleyLocs(iLowest) - 1;

    windowIdx = iMinimum-windowHalfSize:iMinimum+windowHalfSize;
    rawRipples(rippleNum, :) = dataLFP(windowIdx, configChannels.ch.ripple(shankNum));
    filtRipples(rippleNum, :) = filtLFP{shankNum}(windowIdx);

    alignedRippleTimes(rippleNum, :) = [iMinimum-windowHalfSize, iMinimum, iMinimum+windowHalfSize];

    % figure
    % filtRipple = filtLFP{shankNum}(rippleIdx);
    % plot(filtRipple); hold on; plot(env/20); 
    % xline(peakLocs(iHighest)); 
    % xline(peakLocs(iHighest) - slack + valleyLocs(iLowest) - 1)
    % title(rippleNum)

end
    


%%%%%%%%%%%%%%%%calculate CSD


windowSize = 0.1;


windowHalfSize = ceil(windowSize/2 * basicData.fsLFP);


if configChannels.probeType == 8  % Lineal 16ch
    shankNums = 1;
    shankChs = 1:16;  % separated by 100 um
    chJump = 1;
elseif configChannels.probeType == 1  % Lineal 16ch
    shankNums = 1;
    shankChs = 1:15;  % separated by 100 um
    chJump = 1;
end   
previousShankChs = shankChs(1:end-chJump*2);
centralShankChs = shankChs(chJump+1:end-chJump);
nextShankChs = shankChs(chJump*2+1:end);

CSD = cell(1, length(shankNums));
CSDStds = cell(1, length(shankNums));
for shankNum = shankNums
    firstShankCh = find(configChannels.shMap == shankNum, 1);

    previousChs = previousShankChs + firstShankCh - 1;
    centralChs = centralShankChs + firstShankCh - 1;
    nextChs = nextShankChs + firstShankCh - 1;

    CSD{shankNum} = 2*dataLFP(:, centralChs) - dataLFP(:, previousChs) - dataLFP(:, nextChs);
    CSDStds{shankNum} = std(CSD{shankNum});
end


CSDs = nan(windowHalfSize*2+1, length(centralShankChs), height(alignedRippleTimes));

for rippleNum = 1:height(alignedRippleTimes)

    rippleShank = rippleAnalysis.shRipS(rippleNum);

    if ismember(rippleShank, shankNums)

        rippleIdx = alignedRippleTimes(rippleNum, 2)-windowHalfSize:alignedRippleTimes(rippleNum, 2)+windowHalfSize;

        CSDs(:, :, rippleNum) = CSD{rippleShank}(rippleIdx, :) ./ CSDStds{rippleShank};

    end

end

rippleCSDs.CSDs = CSDs;
rippleCSDs.previousShankChs = previousShankChs;
rippleCSDs.centralShankChs = centralShankChs;
rippleCSDs.nextShankChs = nextShankChs;
rippleCSDs.shankNums = shankNums;
rippleCSDs.t = (-windowHalfSize:windowHalfSize)/basicData.fsLFP;
rippleCSDs.windowHalfSize = windowHalfSize;

%%%%%%%%SAVE CSD VALUES

% rad_ch = configChannels.ch.rad;
% slm_ch = configChannels.ch.slm;
% 
idx_so = find(rippleCSDs.centralShankChs == so_ch);
idx_pyr = find(rippleCSDs.centralShankChs == pyr_ch);
idx_rad = find(rippleCSDs.centralShankChs == rad_ch);
idx_slm = find(rippleCSDs.centralShankChs == slm_ch);

if isempty(idx_rad) || isempty(idx_slm)
    error('Rad or SLM channel not found in centralShankChs')
end
CSD_so = squeeze(CSDs(:, idx_so, :));
CSD_pyr = squeeze(CSDs(:, idx_pyr, :));
CSD_rad = squeeze(CSDs(:, idx_rad, :));  % 251 × nRipples
CSD_slm = squeeze(CSDs(:, idx_slm, :));  % 251 × nRipples
% Media temporal por ripple
mean_so = mean(CSD_so, 1);
mean_pyr = mean(CSD_pyr, 1);
mean_rad = mean(CSD_rad, 1);
mean_slm = mean(CSD_slm, 1);

% Máximo por ripple
max_so = max(CSD_so, [], 1);
max_pyr = max(CSD_pyr, [], 1);
max_rad = max(CSD_rad, [], 1);
max_slm = max(CSD_slm, [], 1);

% Mínimo por ripple
min_so = min(CSD_so, [], 1);
min_pyr = min(CSD_pyr, [], 1);
min_rad = min(CSD_rad, [], 1);
min_slm = min(CSD_slm, [], 1);

rippleCSDs.so.values = CSD_so;
rippleCSDs.so.mean   = mean_so;
rippleCSDs.so.max    = max_so;
rippleCSDs.so.min    = min_so;

rippleCSDs.pyr.values = CSD_pyr;
rippleCSDs.pyr.mean   = mean_pyr;
rippleCSDs.pyr.max    = max_pyr;
rippleCSDs.pyr.min    = min_pyr;

rippleCSDs.rad.values = CSD_rad;
rippleCSDs.rad.mean   = mean_rad;
rippleCSDs.rad.max    = max_rad;
rippleCSDs.rad.min    = min_rad;

rippleCSDs.slm.values = CSD_slm;
rippleCSDs.slm.mean   = mean_slm;
rippleCSDs.slm.max    = max_slm;
rippleCSDs.slm.min    = min_slm;

save(fullfile(sessionPath, 'rippleCSDs.mat'), 'rippleCSDs')
% rippleIdx = 6;
% 
% csd_event = squeeze(CSDs(:,:,rippleIdx));  % 251 x 14
% 
% figure
% imagesc(csd_event')   % transponemos para que canales estén en Y
% axis xy               % para que el canal 1 esté arriba
% colormap(jet)         % azul → rojo
% colorbar
% 
% xlabel('Time')
% ylabel('Channel')
% title(['CSD Ripple ' num2str(rippleIdx)])

windowHalfSize = rippleCSDs.windowHalfSize;
rippleCSDs.t = rippleCSDs.t * 1000;

figure('Position', [1, 1, min(600*length(rippleCSDs.shankNums), 1920), 1080])
pad = 10;

for iShank = 1:length(rippleCSDs.shankNums)

    subplot(1, length(rippleCSDs.shankNums), iShank)
    hold on

    set(gca, 'YDir', 'reverse')

    shankNum = rippleCSDs.shankNums(iShank);

    shankChs = union(union(rippleCSDs.previousShankChs, rippleCSDs.centralShankChs), rippleCSDs.nextShankChs);

    chs = shankChs + find(configChannels.shMap == shankNum, 1) - 1;

    shankRippleNums = find(rippleAnalysis.shRipS == shankNum);
    LFPs = nan(size(rippleCSDs.CSDs, 1), length(chs), length(shankRippleNums));
    for iShankRipple = 1:length(shankRippleNums)
        shankRippleNum = shankRippleNums(iShankRipple);
        rippleIdx = alignedRippleTimes(shankRippleNum, 2)-windowHalfSize:alignedRippleTimes(shankRippleNum, 2)+windowHalfSize;
        LFPs(:, :, iShankRipple) = dataLFP(rippleIdx, chs);
    end


    meanCSD = mean(rippleCSDs.CSDs(:, :, rippleAnalysis.shRipS == shankNum), 3, 'omitnan');

    im = imagesc(rippleCSDs.t, find(shankChs == rippleCSDs.centralShankChs(1), 1):find(shankChs == rippleCSDs.centralShankChs(end), 1), meanCSD');
    im.AlphaData = ~isnan(meanCSD');
    clim([-8 5])
    colorbar

    % clim([prctile(meanCSD, 5, 'all'), prctile(meanCSD, 95, 'all')])

    meanLFP = mean(LFPs, 3, 'omitnan');
    normDivisor = max(abs(meanLFP), [], 'all') / 2;
    for iCh = 1:length(chs)
        plot(rippleCSDs.t, -meanLFP(:, iCh) / normDivisor + iCh, 'k')
    end

    xlim([rippleCSDs.t(1), rippleCSDs.t(end)])
    ylim([-2, length(chs)+2])

    areas = {'so', 'pyr', 'rad', 'slm'};

    for areaNum = 1:length(areas)
        if isfield(configChannels.ch, areas{areaNum}) && ~all(isnan(configChannels.ch.(areas{areaNum}))) && ~isnan(configChannels.ch.(areas{areaNum})(shankNum))
            areaCh = configChannels.ch.(areas{areaNum})(shankNum);
            if areaNum ==1
                areaCh = so_ch;
            end
            if areaNum ==2
                areaCh = pyr_ch;
            end
            if areaNum ==3
                areaCh = rad_ch;
            end
            if areaNum ==4
                areaCh = slm_ch;
            end
            % % Find the indices where the value would fit
            % idx = find(chs <= areaCh, 1, 'last');
            % nextIdx = idx + 1;
            % 
            % % Calculate the fractional position
            % if idx < length(chs)
            %     fraction = (areaCh - chs(idx)) / (chs(nextIdx) - chs(idx));
            %     position = idx + fraction;
            % else
            %     position = idx;
            % end
            position = areaCh;
            text(rippleCSDs.t(1)-pad, position, upper(areas{areaNum}))
        end
    end

    yticks(1:length(chs))

    title(sprintf('shank %d (%d ripples)', shankNum, length(shankRippleNums)), 'Interpreter', 'none')

    xlabel('Time (ms)')

    yLim = ylim;

    yyaxis right
    ylim(yLim)
    set(gca, 'YDir', 'reverse')
    yticks(1:length(chs))
    yticklabels(chs)


end
% Session name
[animalPath, sessionName] = fileparts(sessionPath);

% Animal name
[~, animalName] = fileparts(animalPath);
sgtitle(sessionName, 'interpreter', 'none')


saveas(gcf, fullfile(sessionPath, [sessionName, 'MeanRippleCSD.png']))



