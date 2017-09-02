function [] = create_figures()

exps = {'ivy2/512_2','ivy2/512_3','ivy2/512_4','ivy2/512_5',...
    'ivy2/512_6','ivy2/512_7','ivy2/512_8','ivy2/512_9','ivy2/512_10',...
    'ivy2/512_11','ivy2/512_12'};

for i = 1:length(exps)
    mkdir(['../figures/' exps{i}], 's');
    
    % loss
    load(['../nets/' exps{i} '/loss.mat']);
    figure;
    plot(loss);
    saveas(gcf, ['../figures/' exps{i} '/loss.png']);
    close;
    
    % gamma2
    load(['../nets/' exps{i} '/config.mat']);
    figure;
    plot(config.Gamma2);
    saveas(gcf, ['../figures/' exps{i} '/gamma2.png']);
    close;
end

for i = 1:length(exps)

    % loss
    load(['../nets/' exps{i} '/loss.mat']);
    disp([exps{i} ' ' num2str(loss(end))]);
end


end

