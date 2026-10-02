function plot_group_comparison(WT, NPC, EFV, yLabelText, figTitle)

figure; hold on;

% ----- Preparar datos
data = [WT(:); NPC(:); EFV(:)];
group = [ ...
    repmat({'WT'},length(WT),1); ...
    repmat({'NPC'},length(NPC),1); ...
    repmat({'EFV'},length(EFV),1)];

groupCat = categorical(group);

% ----- Boxplot
boxplot(data, groupCat, 'Symbol','')

% ----- Añadir puntos individuales
x_numeric = double(groupCat);
scatter(x_numeric, data, 15, 'k', 'filled', ...
    'jitter','on','jitterAmount',0.15, ...
    'MarkerFaceAlpha',0.4)

ylabel(yLabelText)
title(figTitle)
set(gca,'FontSize',12)
box off

% ----- ANOVA
[p,tbl,stats] = anova1(data, group, 'off');

% ----- Posthoc Tukey
c = multcompare(stats,'Display','off');

% ----- Añadir estrellas
yMax = max(data);
yMin = min(data);
yRange = yMax - yMin;
yStep = yRange * 0.08;
currentY = yMax + yStep;

for i = 1:size(c,1)

    pval = c(i,6);

    if pval < 0.05

        x1 = c(i,1);
        x2 = c(i,2);

        plot([x1 x1 x2 x2], ...
             [currentY currentY+yStep currentY+yStep currentY], ...
             'k','LineWidth',1.5)

        if pval < 0.001
            stars = '***';
        elseif pval < 0.01
            stars = '**';
        else
            stars = '*';
        end

        text(mean([x1 x2]), currentY + yStep*1.2, stars, ...
            'HorizontalAlignment','center','FontSize',14)

        currentY = currentY + yStep*1.8;
    end
end

end
