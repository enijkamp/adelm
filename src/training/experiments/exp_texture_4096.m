function [] = exp_texture_4096()

rng(123);

% config
img_name = 'ivy2';
img_size = 4096;
patch_size = 32;

% setup
use_gpu = 1;
compile_convnet = 0;

root = setup_path();
setup_convnet(use_gpu, compile_convnet);

% prep
prefix = [img_name '/' num2str(img_size) '_1/'];
[config, net1] = coopnet_config(root);
config = prep_images(config, [root 'data/' img_name '/' num2str(img_size) '/'], patch_size);
config = prep_dirs(config, prefix);
config.use_gpu = use_gpu;
config.nIteration = 500;     

% sampling parameters
config.num_syn = 32;
% descriptor net1 parameters
config.Delta = 0.3;
config.Gamma = [0.0005*ones(1,100), 0.00005*ones(1,200), 0.000005*ones(1,100)];     % step-wise
config.refsig = 1;                                            
config.T = 40;                                                                                                                
% generator net2 parameters
config.Delta2 = 0.3;
config.Gamma2 = [0.00035*ones(1,100), 0.000035*ones(1,200), 0.0000035*ones(1,100)];    % step-wise
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;

% learn
learn_dual_net(config, net1);

end
