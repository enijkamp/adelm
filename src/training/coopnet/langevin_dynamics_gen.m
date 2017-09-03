function z = langevin_dynamics_gen(config, net, z, syn_mat)
% the input syn_mat should be a 4-D matrix

for t = 1:config.T
    res = vl_gan(net, z, syn_mat, [], 'conserveMemory', 1, 'cudnn', 1);
    delta_log = res(1).dzdx / config.refsig2 / config.refsig2 - z;
    z = z + config.Delta2 * config.Delta2 / 2 * delta_log;
    %z = z + config.Delta2 * gpuArray(randn(size(z), 'single'));
    clear res;  
end
end