function  [net1,net2,gen_mat,syn_mat,z] = process_epoch_dual(opts, epoch, net1, net2, config)

%% TODO is this fast? we don't want to move nets back and forth between cpu/gpu (en)
if config.use_gpu
    net1 = vl_simplenn_move(net1, 'gpu') ;
    net2 = vl_simplenn_move(net2, 'gpu') ;
end

fprintf('Training: epoch %02d', epoch) ;
fprintf('\n');

% randomize order of training images
imdb = config.imdb;
train_order = randperm(size(imdb,4));
batchNum = 0;
epoch_time = tic;

for t = 1:config.batchSize:size(imdb,4)
    
    batchNum = batchNum+1;
    fprintf('Epoch %d of %d, Batch %d of %d \n', epoch,config.nIteration,batchNum, ceil(size(imdb,4)/config.batchSize));
    batchTime = tic;

    batchStart = t;
    batchEnd = min(t+config.batchSize-1, size(imdb,4));
    batch = train_order(batchStart : batchEnd);
    im = imdb(:,:,:,batch);   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1: Generate synthesis from z
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % G0: generate Xi
    z = randn([config.z_sz, config.num_syn], 'single');

    % D1: generate Yi
    z = toGpuArray(z, config.use_gpu);
    syn_mat = vl_gan(net2, z, [], [],...
            'accumulate', 0, ...
            'disableDropout', 0, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
    syn_mat = syn_mat(end).x; 

    %% TODO cleanup (en)
    % syn_mat = floor(128*(syn_mat+1))-config.mean_im;
    syn_mat = floor(128*(syn_mat+1)) - repmat(config.mean_im, 1, 1, 1, config.num_syn);
    gen_mat = gather(syn_mat);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 2: (1) Update generator mats by descriptor net, (2) Update z
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % synthesize image according to current weights 
    % D1: generate y wave i
    if config.use_gpu
        syn_mat = langevin_dynamics_fast(config, net1, syn_mat);
    else
        for syn_ind = 1:config.num_syn
            syn_mat(:,:,:,syn_ind) = langevin_dynamics(config, net1, syn_mat(:,:,:,syn_ind));
        end
    end
    
    % G1: Y wave - syn_mat
    % Xj - z
    % run langevin IG steps to update z
    if config.infer_z
        z = langevin_dynamics_gen(config, net2, z, syn_mat);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3: Learn net1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    im = toGpuArray(im, config.use_gpu);
    dydz1 = toGpuArray(zeros(config.dydz_sz1, 'single'), config.use_gpu);
    dydz1 = repmat(dydz1,1,1,1,size(im,4));
    
    res1 = [];
    res1 = vl_simplenn(net1, im, dydz1, res1, ...
        'accumulate', 0, ...
        'disableDropout', 0, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn);

    dydz_syn = zeros(config.dydz_sz1, 'single');
    dydz_syn = toGpuArray(repmat(dydz_syn,1,1,1,config.num_syn), config.use_gpu);
    
    res_syn = [];
    res_syn = vl_simplenn(net1, syn_mat, dydz_syn, res_syn, ...
        'accumulate', 0, ...
        'disableDropout', 0, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn);

    %% TODO cleanup (en)
    % syn_mat_2 = max(min(syn_mat+config.mean_im,255.99),0.01)/128 - 1;
    syn_mat_2 = max(min(syn_mat+repmat(config.mean_im, 1, 1, 1, config.num_syn),255.99),0.01)/128 - 1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 4: Learn net2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    res2 = [];
    res2 = vl_gan(net2, z, syn_mat_2, res2, ...
        'accumulate', 0, ...
        'disableDropout', 0, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;

    net1 = accumulate_gradients1(opts, config.Gamma(epoch), length(batch), net1, res1, res_syn, config);
    net2 = accumulate_gradients2(opts, config.Gamma2(epoch), length(batch), net2, res2, config);     
    
    config.real_ref = std(z(:));  
    fprintf('max inferred z is %.2f, min inferred z is %.2f, and std is %.2f\n', max(z(:)), min(z(:)), config.real_ref);
    
    batchTime = toc(batchTime);
    speed = 1/batchTime;
    fprintf(' %Time: .2f s (%.1f data/s)\n', batchTime, speed);
end

if config.use_gpu
    net1 = vl_simplenn_move(net1, 'cpu') ;
    net2 = vl_simplenn_move(net2, 'cpu') ;
end

% print learning statistics
epoch_time = toc(epoch_time) ;
speed = config.num_syn/epoch_time;

fprintf(' %.2f s (%.1f data/s)', epoch_time, speed) ;
fprintf('\n') ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients1(opts, lr, batchSize, net, res, res_syn, config)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets1;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;

        if isfield(net.layers{l}, 'weights')
            
            gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                        (1 / config.num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
                    
            if max(abs(gradient_dzdw(:))) > 20 %10
                gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * 20;
            end
            
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('\n layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end
 

% -------------------------------------------------------------------------
function [net, res] = accumulate_gradients2(opts, lr, batchSize, net, res, config)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets2;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        if isfield(net.layers{l}, 'weights')
            % gradient descent
            % gradient_dzdw = (1 / config.s / config.s) * res(l).dzdw{j};
            gradient_dzdw = (1 / batchSize) * (1 / config.s / config.s) * res(l).dzdw{j};
            
            max_val = max(abs(gradient_dzdw(:)));
            
            if max_val > config.cap2
                gradient_dzdw = gradient_dzdw / max_val * config.cap2;
            end
  
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('Net2: layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end

