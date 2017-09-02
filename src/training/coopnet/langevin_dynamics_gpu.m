function syn_mat = langevin_dynamics_fast(config, net, syn_mat)

% syn_mat = gpuArray(syn_mat);
% net = vl_simplenn_move(net, 'gpu') ;

numImages = size(syn_mat, 4);

dydz = gpuArray(ones(config.dydz_sz1, 'single'));
% dydz(net.filterSelected) = net.selectedLambdas;
dydz = repmat(dydz, 1, 1, 1, numImages);


for t = 1:config.T
%     fprintf('Langevin dynamics sampling iteration %d\n', t);
    % forward-backward to compute df/dI
%     N_gaussian = gpuArray(randn(size(syn_mat), 'single'));
    res = vl_simplenn(net, syn_mat, dydz, [], 'conserveMemory', 1, 'cudnn', 1);
    
    % part1: derivative on f(I; w)  part2: gaussian I
    syn_mat = syn_mat + config.Delta * config.Delta /2 * res(1).dzdx ...  
        - config.Delta /2 /config.refsig / config.refsig * syn_mat;
    
    % part3: white noise N(0, 1)
    syn_mat = syn_mat + config.Delta * gpuArray(randn(size(syn_mat), 'single'));
    clear res;
end

% syn_mat = gather(syn_mat);
end