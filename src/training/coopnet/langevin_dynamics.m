function syn_mat = langevin_dynamics(config, net, syn_mat)
% the input syn_mat should be a 3-D matrix
dydz = zeros(net.dydz_sz, 'single');
dydz(net.filterSelected) = net.selectedLambdas;

T = config.T;

for t = 1:T    
    % Leapfrog half-step
    res = vl_simplenn(net, syn_mat, dydz, [], 'conserveMemory', 1, 'cudnn', false);

    % part1: derivative on f(I; w)  part2: gaussian I
    syn_mat = syn_mat + config.Delta * config.Delta /2 * res(1).dzdx ...  
        - config.Delta * config.Delta /2 /config.refsig /config.refsig* syn_mat;
    
    % part3: white noise N(0, 1)
    syn_mat = syn_mat + config.Delta * randn(size(syn_mat), 'single');
    
    clear res;
end


end