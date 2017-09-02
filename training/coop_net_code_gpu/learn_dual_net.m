function [net1,net2,gen_mats,syn_mats] = learn_dual_net(config, net1)
    learningTime = tic;
    
    % generator
    net1 = add_bottom_filters(net1, 1:3, config);
    net1.mean_im = config.mean_im;
    
    % descriptor
    net2 = frame_gan_params(config);
 
    % generator net config
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
   
    %% GPU (en) %%
    if config.use_gpu
        net2 = vl_simplenn_move(net2, 'gpu');
        res = vl_gan(net2, gpuArray(z));
        net2 = vl_simplenn_move(net2, 'cpu');
    else
        res = vl_gan_cpu(net2, z);
    end

    
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
    
    % descriptor net config
    img = randn([config.im_size,config.im_size,3],'single');
    
    %% GPU (en) %%
    if config.use_gpu
        net1 = vl_simplenn_move(net1, 'gpu') ;
        res = vl_simplenn(net1, gpuArray(img));
        net1 = vl_simplenn_move(net1, 'cpu');
    else
        res = vl_simplenn(net1, img);
    end

    
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
    
    % training options
    
    opts.conserveMemory = true ;
    opts.backPropDepth = +inf ;
    opts.sync = false ;
    opts.prefetch = false;
    opts.cudnn = true ;
    opts.weightDecay = 0.0001 ;
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

    %% GPU (en) %%
    if config.use_gpu
        numGpus = 1;
        gpuDevice(numGpus);
    end
    
    % -------------------------------------------------------------------------
    %                           Train and validate
    % -------------------------------------------------------------------------

    delete([config.gen_im_folder,'*.png']);
    delete([config.syn_im_folder,'*.png']);
    
    save([config.trained_folder,'config.mat'],'config');
    
    loss = zeros(config.nIteration, 1);
    for epoch=1:config.nIteration
        [net1, net2, gen_mats, syn_mats, z] = process_epoch_dual(opts,epoch,net1,net2,config);
        loss(epoch) = compute_loss(opts, syn_mats, net2, z, config);
        save([config.trained_folder,'loss.mat'],'loss');
        disp(['Loss: ', num2str(loss(epoch))]);
    end
    
    learningTime = toc(learningTime);
    hrs = floor(learningTime / 3600);
    learningTime = mod(learningTime, 3600);
    mins = floor(learningTime / 60);
    secds = mod(learningTime, 60);
    fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

end

function loss = compute_loss(opts, syn_mat, net2_cpu, z, config)
%% GPU (en) %%
if config.use_gpu
    net2 = vl_simplenn_move(net2_cpu, 'gpu');
    res = [];
    res = vl_gan(net2, gpuArray(z), gpuArray(syn_mat), res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    loss = gather( mean(reshape(sqrt((res(end).x - syn_mat).^2), [], 1)));
else
    res = [];
    res = vl_gan_cpu(net2_cpu, z, syn_mat, res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    loss = gather( mean(reshape(sqrt((res(end).x - syn_mat).^2), [], 1)));
end
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