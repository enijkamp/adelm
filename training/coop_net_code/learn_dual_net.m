function [net1,net2,gen_mats,syn_mats] = learn_dual_net(net1,net2,config,fix_des,fix_gen)
    if nargin < 4 || isempty(fix_des), fix_des = 0; end
    if nargin < 5 || isempty(fix_gen), fix_gen = 0; end
    learningTime = tic;
    % read in config
    if nargin<3 || isempty(config), config = train_coop_config; end
    
    if nargin < 2 || isempty(net2)
        net2 = frame_gan_params();
    end

    if nargin < 1 || isempty(net1)
        [~,net1] = train_coop_config;
        net1 = add_bottom_filters(net1, 1:3,config);
        net1.mean_im = config.mean_im;
    end
 
    %generator net config
    config.z_sz = [1, 1, size(net2.layers{1}.weights{1}, 4)];
    net2.z_sz = config.z_sz;
    config.dydz_sz2 = [config.z_sz(1:2), 1];
    for l = 1:numel(net2.layers)
        if strcmp(net2.layers{l}.type, 'convt')
            f_sz = size(net2.layers{l}.weights{1});
            crops = [net2.layers{l}.crop(1)+net2.layers{l}.crop(2), ...
                net2.layers{l}.crop(3)+net2.layers{l}.crop(4)];
            config.dydz_sz2(1:2) = net2.layers{l}.upsample.*(config.dydz_sz2(1:2) - 1) ...
                + f_sz(1:2) - crops;
        end
    end
    net2.dydz_sz = config.dydz_sz2;

    z = randn(config.z_sz, 'single');
   
    res = vl_gan_cpu(net2, z);
    %disp(size(res(end).x));
    net2.numFilters = zeros(1, length(net2.layers));
    for l = 1:length(net2.layers)
        if isfield(net2.layers{l}, 'weights')
            sz = size(res(l+1).x);
            net2.numFilters(l) = sz(1) * sz(2);
        end
    end

    config.layer_sets2 = numel(net2.layers):-1:1;

    net2.normalization.imageSize = config.dydz_sz2;
    net2.normalization.averageImage = zeros(config.dydz_sz2, 'single');
    config.sx = config.dydz_sz2(1);
    config.sy = config.dydz_sz2(2);
    clear z;
    
    %descriptor net config
    img = randn([config.im_size,config.im_size,3],'single');
    res = vl_simplenn(net1, img);
    config.dydz_sz1 = size(res(end).x);
    net1.numFilters = zeros(1, length(net1.layers));
    for l = 1:length(net1.layers)
        if isfield(net1.layers{l}, 'weights')
           sz = size(res(l+1).x);
          net1.numFilters(l) = sz(1) * sz(2);
        end
    end
    net1.dydz_sz = config.dydz_sz1;

    config.layer_sets1 = numel(net1.layers):-1:1;

    clear res;
    clear img;
    
    %training options
    
    opts.conserveMemory = true ;
    opts.backPropDepth = +inf ;
    opts.sync = false ;
    opts.prefetch = false;
    opts.cudnn = true ;
    opts.weightDecay = 0.0001 ; %0.0001
    opts.momentum = 0.5;

    net1 = initialize_momentum(net1);
    net2 = initialize_momentum(net2);

    for i=1:numel(net1.layers)
        if isfield(net1.layers{i}, 'weights')
            J = numel(net1.layers{i}.weights);
            for j=1:J
                net1.layers{i}.momentum{j} = zeros(size(net1.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net1.layers{i}, 'learningRate')
                net1.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net1.layers{i}, 'weightDecay')
                net1.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end

    net1.filterSelected = 1:prod(config.dydz_sz1);
    net1.selectedLambdas = ones(1, prod(config.dydz_sz1), 'single');

    % -------------------------------------------------------------------------
    %                           Train and validate
    % -------------------------------------------------------------------------

    delete([config.gen_im_folder,'*.png']);
    delete([config.syn_im_folder,'*.png']);
    for epoch=1:config.nIteration
        [net1_out,net2_out,gen_mats,syn_mats] = process_epoch_dual(opts,epoch,net1,net2,config);     
        if ~fix_des, net1 = net1_out; end
        if ~fix_gen, net2 = net2_out; end
    end
    
    learningTime = toc(learningTime);
    hrs = floor(learningTime / 3600);
    learningTime = mod(learningTime, 3600);
    mins = floor(learningTime / 60);
    secds = mod(learningTime, 60);
    fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);
end


function net = initialize_momentum(net)
for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        for j=1:J
            net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
        end
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = ones(1, J, 'single') ;
        end
    end
end
end