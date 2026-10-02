%% ================== SETTINGS ==================
baseDir = 'E:\Violeta\ANALISIS';  % Carpeta con los animales
animalDirs = dir(baseDir);
animalDirs = animalDirs([animalDirs.isdir] & ~ismember({animalDirs.name},{'.','..'}));
% Colores para cada grupo (RGB/256)
groupColors = [ ...
   76 153 102;   % WT 
   70 120 180;   % NPC 
   150 110 170];  % EFV 
groupColors = groupColors / 256;
groupNames = {'WT','NPC','EFV'};
% allSo = [];
% allPyr = [];
% allRad = [];
% allSlm = [];
% groupIdxSo = [];
% groupIdxPyr = [];
% groupIdxRad = [];
% groupIdxSlm = [];

% Initialize arrays
soValues = [];
pyrValues = [];
radValues = [];
slmValues = [];
soGroupLabels = {};
pyrGroupLabels = {};
radGroupLabels = {};
slmGroupLabels = {};

%% Load all sessions for each animal
for iAnimal = 1:length(animalDirs)
    animalName = animalDirs(iAnimal).name;
    
    % Detect group based on animal name
    if contains(animalName,'WT')
        grp = 'WT';
    elseif contains(animalName,'NPC')
        grp = 'NPC';
    elseif contains(animalName,'EFV')
        grp = 'EFV';
    else
        continue % skip if animal name does not match
    end
    
    % All sessions for this animal
    sessionDirs = dir(fullfile(baseDir, animalName));
    sessionDirs = sessionDirs([sessionDirs.isdir] & ~ismember({sessionDirs.name},{'.','..'}));
    
    for iSess = 1:length(sessionDirs)
        sessionPath = fullfile(baseDir, animalName, sessionDirs(iSess).name);
        
        % Skip if rippleCSDs file does not exist
        csdFile = fullfile(sessionPath,'rippleCSDs.mat');
        if ~exist(csdFile,'file'); continue; end
        
        S = load(csdFile,'rippleCSDs');
        
        % Concatenate all ripples
        nTime = size(S.rippleCSDs.rad.values,1);   % 251
        centerIdx = ceil(nTime/2);
        halfWindow = 25;
        idx = centerIdx-halfWindow+1 : centerIdx+halfWindow;
        centralMeanSo = mean(S.rippleCSDs.so.values(idx,:), 1);
        soValues = [soValues; centralMeanSo'];
        centralMeanPyr = mean(S.rippleCSDs.pyr.values(idx,:), 1);
        pyrValues = [pyrValues; centralMeanPyr'];
        centralMeanRad = mean(S.rippleCSDs.rad.values(idx,:), 1);
        radValues = [radValues; centralMeanRad'];
        %radValues = [radValues; S.rippleCSDs.rad.mean(:)];
        centralMeanSLM = mean(S.rippleCSDs.slm.values(idx,:), 1);
        slmValues = [slmValues; centralMeanSLM'];
        % slmValues = [slmValues; S.rippleCSDs.slm.mean(:)];
        
        % Assign group labels for each ripple
        soGroupLabels = [soGroupLabels; repmat({grp}, numel(S.rippleCSDs.so.mean),1)];
        pyrGroupLabels = [pyrGroupLabels; repmat({grp}, numel(S.rippleCSDs.pyr.mean),1)];
        radGroupLabels = [radGroupLabels; repmat({grp}, numel(S.rippleCSDs.rad.mean),1)];
        slmGroupLabels = [slmGroupLabels; repmat({grp}, numel(S.rippleCSDs.slm.mean),1)];
        % groupLabels = [groupLabels; repmat({grp}, numel(S.rippleCSDs.rad.mean),1)];
    end
end


%% Plot SO CSD
figure; hold on;

groupOrder = {'WT','NPC','EFV'};
% groupLabelsCat = categorical(groupLabels, groupOrder, 'Ordinal',true);
groupLabelsCat = categorical(soGroupLabels, groupOrder, 'Ordinal',true);
data = soValues;   % <-- ahora usamos SLM


% Boxplot con orden fijo
boxplot(data, groupLabelsCat, 'Colors','k','Symbol','');
set(findobj(gca,'Tag','Box'),'LineWidth',1.5);


% Scatter de todos los ripples
for iG = 1:length(groupOrder)
    idx = groupLabelsCat == groupOrder{iG};
    scatter( repmat(iG,sum(idx),1) + 0.15*randn(sum(idx),1), ...
             data(idx), 10, ...
             'MarkerFaceColor', groupColors(iG,:), ...
             'MarkerEdgeColor','k', ...
             'MarkerFaceAlpha',0.4);
end

ylabel('CSD So');
title('CSD So by Group');

%% --- KRUSKAL WALLIS ---
[p_kw, tbl_kw, stats_kw] = kruskalwallis(data, groupLabelsCat, 'off');

fprintf('\n--- Kruskal-Wallis (SLM) ---\n');
fprintf('Chi^2 = %.4f | df = %.0f | p = %.4g\n', ...
        tbl_kw{2,5}, tbl_kw{2,3}, p_kw);

%% --- MULTCOMPARE ---
c = multcompare(stats_kw,'Display','off');

fprintf('\nPairwise comparisons (SLM):\n');
fprintf('Group1\tGroup2\tp-value\n');

for i = 1:size(c,1)
    grp1 = stats_kw.gnames{c(i,1)};
    grp2 = stats_kw.gnames{c(i,2)};
    fprintf('%s\t%s\t%.4g\n', grp1, grp2, c(i,6));
end

%% --- ESTRELLAS ---
yMax = max(data);
yMin = min(data);
yRange = yMax - yMin;

ylim([yMin, yMax + 0.25*yRange]);

starY = yMax + 0.05*yRange;
starSpacing = 0.07*yRange;

for i = 1:size(c,1)

    pVal = c(i,6);
    if pVal < 0.05

        x1 = c(i,1);
        x2 = c(i,2);

        if pVal < 0.001
            starText = '***';
        elseif pVal < 0.01
            starText = '**';
        else
            starText = '*';
        end

        plot([x1 x2],[starY starY],'k','LineWidth',1.5)
        text(mean([x1 x2]), starY + 0.01*yRange, starText, ...
             'HorizontalAlignment','center','FontSize',14)

        starY = starY + starSpacing;
    end
end
saveas(gcf, fullfile(baseDir,'soCSD.png'))

%% Plot PYR CSD
figure; hold on;

groupOrder = {'WT','NPC','EFV'};
% groupLabelsCat = categorical(groupLabels, groupOrder, 'Ordinal',true);

data = pyrValues;   % <-- ahora usamos PYR
groupLabelsCat = categorical(pyrGroupLabels, groupOrder, 'Ordinal',true);



% Boxplot con orden fijo
boxplot(data, groupLabelsCat, 'Colors','k','Symbol','');
set(findobj(gca,'Tag','Box'),'LineWidth',1.5);


% Scatter de todos los ripples
for iG = 1:length(groupOrder)
    idx = groupLabelsCat == groupOrder{iG};
    scatter( repmat(iG,sum(idx),1) + 0.15*randn(sum(idx),1), ...
             data(idx), 10, ...
             'MarkerFaceColor', groupColors(iG,:), ...
             'MarkerEdgeColor','k', ...
             'MarkerFaceAlpha',0.4);
end

ylabel('CSD Pyr');
title('CSD Pyr by Group');

%% --- KRUSKAL WALLIS ---
[p_kw, tbl_kw, stats_kw] = kruskalwallis(data, groupLabelsCat, 'off');

fprintf('\n--- Kruskal-Wallis (SLM) ---\n');
fprintf('Chi^2 = %.4f | df = %.0f | p = %.4g\n', ...
        tbl_kw{2,5}, tbl_kw{2,3}, p_kw);

%% --- MULTCOMPARE ---
c = multcompare(stats_kw,'Display','off');

fprintf('\nPairwise comparisons (SLM):\n');
fprintf('Group1\tGroup2\tp-value\n');

for i = 1:size(c,1)
    grp1 = stats_kw.gnames{c(i,1)};
    grp2 = stats_kw.gnames{c(i,2)};
    fprintf('%s\t%s\t%.4g\n', grp1, grp2, c(i,6));
end

%% --- ESTRELLAS ---
yMax = max(data);
yMin = min(data);
yRange = yMax - yMin;

ylim([yMin, yMax + 0.25*yRange]);

starY = yMax + 0.05*yRange;
starSpacing = 0.07*yRange;

for i = 1:size(c,1)

    pVal = c(i,6);
    if pVal < 0.05

        x1 = c(i,1);
        x2 = c(i,2);

        if pVal < 0.001
            starText = '***';
        elseif pVal < 0.01
            starText = '**';
        else
            starText = '*';
        end

        plot([x1 x2],[starY starY],'k','LineWidth',1.5)
        text(mean([x1 x2]), starY + 0.01*yRange, starText, ...
             'HorizontalAlignment','center','FontSize',14)

        starY = starY + starSpacing;
    end
end
saveas(gcf, fullfile(baseDir,'pyrCSD.png'))






%% Plot Radial CSD
figure; hold on;

groupOrder = {'WT','NPC','EFV'};
%groupLabelsCat = categorical(groupLabels, groupOrder, 'Ordinal',true);
groupLabelsCat = categorical(radGroupLabels, groupOrder, 'Ordinal',true);

data = radValues;


% Boxplot con orden fijo
boxplot(data, groupLabelsCat, 'Colors','k','Symbol','');
set(findobj(gca,'Tag','Box'),'LineWidth',1.5);


% Scatter de todos los ripples
for iG = 1:length(groupOrder)
    idx = groupLabelsCat == groupOrder{iG};
    scatter( repmat(iG,sum(idx),1) + 0.15*randn(sum(idx),1), ...
             data(idx), 10, ...
             'MarkerFaceColor', groupColors(iG,:), ...
             'MarkerEdgeColor','k', ...
             'MarkerFaceAlpha',0.4);
end

ylabel('CSD Radial');
title('CSD Radial by Group');

%% --- KRUSKAL WALLIS ---
[p_kw, tbl_kw, stats_kw] = kruskalwallis(data, groupLabelsCat, 'off');

fprintf('\n--- Kruskal-Wallis (Radial) ---\n');
fprintf('Chi^2 = %.4f | df = %.0f | p = %.4g\n', ...
        tbl_kw{2,5}, tbl_kw{2,3}, p_kw);

%% --- MULTCOMPARE ---
c = multcompare(stats_kw,'Display','off');

fprintf('\nPairwise comparisons:\n');
fprintf('Group1\tGroup2\tp-value\n');

for i = 1:size(c,1)
    grp1 = stats_kw.gnames{c(i,1)};
    grp2 = stats_kw.gnames{c(i,2)};
    fprintf('%s\t%s\t%.4g\n', grp1, grp2, c(i,6));
end

%% --- ESTRELLAS ---
yMax = max(data);
yMin = min(data);
yRange = yMax - yMin;

% espacio extra arriba para estrellas
ylim([yMin, yMax + 0.25*yRange]);
starY = yMax + 0.05*yRange;
starSpacing = 0.07*yRange;

for i = 1:size(c,1)

    pVal = c(i,6);
    if pVal < 0.05

        x1 = c(i,1);   % ahora coinciden 1=WT, 2=NPC, 3=EFV
        x2 = c(i,2);

        if pVal < 0.001
            starText = '***';
        elseif pVal < 0.01
            starText = '**';
        else
            starText = '*';
        end

        plot([x1 x2],[starY starY],'k','LineWidth',1.5)
        text(mean([x1 x2]), starY + 0.01*yRange, starText, ...
             'HorizontalAlignment','center','FontSize',14)

        starY = starY + starSpacing;
    end
end

saveas(gcf, fullfile(baseDir,'RadCSD.png'))

%% Plot SLM CSD
figure; hold on;

groupOrder = {'WT','NPC','EFV'};
%groupLabelsCat = categorical(groupLabels, groupOrder, 'Ordinal',true);
groupLabelsCat = categorical(slmGroupLabels, groupOrder, 'Ordinal',true);

data = slmValues;   % <-- ahora usamos SLM


% Boxplot con orden fijo
boxplot(data, groupLabelsCat, 'Colors','k','Symbol','');
set(findobj(gca,'Tag','Box'),'LineWidth',1.5);


% Scatter de todos los ripples
for iG = 1:length(groupOrder)
    idx = groupLabelsCat == groupOrder{iG};
    scatter( repmat(iG,sum(idx),1) + 0.15*randn(sum(idx),1), ...
             data(idx), 10, ...
             'MarkerFaceColor', groupColors(iG,:), ...
             'MarkerEdgeColor','k', ...
             'MarkerFaceAlpha',0.4);
end

ylabel('CSD SLM');
title('CSD SLM by Group');

%% --- KRUSKAL WALLIS ---
[p_kw, tbl_kw, stats_kw] = kruskalwallis(data, groupLabelsCat, 'off');

fprintf('\n--- Kruskal-Wallis (SLM) ---\n');
fprintf('Chi^2 = %.4f | df = %.0f | p = %.4g\n', ...
        tbl_kw{2,5}, tbl_kw{2,3}, p_kw);

%% --- MULTCOMPARE ---
c = multcompare(stats_kw,'Display','off');

fprintf('\nPairwise comparisons (SLM):\n');
fprintf('Group1\tGroup2\tp-value\n');

for i = 1:size(c,1)
    grp1 = stats_kw.gnames{c(i,1)};
    grp2 = stats_kw.gnames{c(i,2)};
    fprintf('%s\t%s\t%.4g\n', grp1, grp2, c(i,6));
end

%% --- ESTRELLAS ---
yMax = max(data);
yMin = min(data);
yRange = yMax - yMin;

ylim([yMin, yMax + 0.25*yRange]);

starY = yMax + 0.05*yRange;
starSpacing = 0.07*yRange;

for i = 1:size(c,1)

    pVal = c(i,6);
    if pVal < 0.05

        x1 = c(i,1);
        x2 = c(i,2);

        if pVal < 0.001
            starText = '***';
        elseif pVal < 0.01
            starText = '**';
        else
            starText = '*';
        end

        plot([x1 x2],[starY starY],'k','LineWidth',1.5)
        text(mean([x1 x2]), starY + 0.01*yRange, starText, ...
             'HorizontalAlignment','center','FontSize',14)

        starY = starY + starSpacing;
    end
end
saveas(gcf, fullfile(baseDir,'slmCSD.png'))