

clear; close all; clc; 

baseDir = 'E:\Violeta\ANALISIS';
animalDirs = dir(baseDir);
animalDirs = animalDirs([animalDirs.isdir]);
animalDirs(ismember({animalDirs.name},{'.','..'})) = [];
cd(baseDir);
% ---- Cargar Excel UNA sola vez ----
channelsTable = readtable('channels.xlsx');
channelsTable.animal  = string(channelsTable.animal);
channelsTable.session = string(channelsTable.session);

for iAnimal = 1:length(animalDirs)

    animalPath = fullfile(animalDirs(iAnimal).folder, animalDirs(iAnimal).name);
    fprintf('\nProcessing animal: %s\n', animalDirs(iAnimal).name);

    sessionDirs = dir(animalPath);
    sessionDirs = sessionDirs([sessionDirs.isdir]);
    sessionDirs(ismember({sessionDirs.name},{'.','..'})) = [];

    for iSess = 1:length(sessionDirs)

        sessionPath = fullfile(sessionDirs(iSess).folder, sessionDirs(iSess).name);
        fprintf('   Session: %s\n', sessionDirs(iSess).name);

        try

            % ---- Buscar canales en Excel ----
            animalName  = string(animalDirs(iAnimal).name);
            sessionName = string(sessionDirs(iSess).name);

            idxRow = find(channelsTable.animal == animalName & ...
                          channelsTable.session == sessionName);

            if isempty(idxRow)
                error('No matching row in channels.xlsx')
            end
            so_ch = channelsTable.so(idxRow);
            pyr_ch = channelsTable.pyr(idxRow);
            rad_ch = channelsTable.rad(idxRow);
            slm_ch = channelsTable.slm(idxRow);

            % ---- Llamar función pasando canales ----
            processRippleSession(sessionPath, so_ch, pyr_ch, rad_ch, slm_ch);

        catch ME
            warning('Error in %s: %s', sessionDirs(iSess).name, ME.message);
        end

    end
end
