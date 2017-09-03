function [] = exp_texture_512_18()

rng(123);

% config
img_name = 'ivy2';
img_size = 512;
patch_size = 32;

% setup
use_gpu = 1;
compile_convnet = 0;

root = setup_path();
setup_convnet(use_gpu, compile_convnet);

%% same as (14), but gradient fix for generator, langevin missing delta fix

% config
[config, net1] = coopnet_config(root);
config.use_gpu = use_gpu;
config.nIteration = 500;                                                  
% sampling parameters
config.num_syn = 32;
% descriptor net1 parameters
config.Delta = 0.002;
config.Gamma = [0.05*ones(1,100), 0.005*ones(1,100), 0.001*ones(1,100), 0.0005*ones(1,100), 0.0001*ones(1,100)];
config.refsig = 0.016;                                            
config.T = 40;                                                                                                                
% generator net2 parameters
config.Delta2 = 0.0000003;
config.Gamma2 = [0.0003*ones(1,100), 0.0001*ones(1,100), 0.00005*ones(1,100), 0.00005*ones(1,100), 0.00001*ones(1,100)] * 10; % (2) lower gamma in beginning, higher in middle 
config.refsig2 = 0.00001;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;
config.infer_z = false;  % disabled inference

% prep
prefix = [img_name '/' num2str(img_size) '_18/'];
config = prep_images(config, [root 'data/' img_name '/' num2str(img_size) '/'], patch_size);
config = prep_dirs(config, prefix);

% learn
learn_dual_net(config, net1);

end
