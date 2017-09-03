function [] = main()
% simple example of training coop-net on CPU

rng(123);

% config
img_name = 'ivy2';
img_size = 64;
patch_size = 32;

% setup
use_gpu = false;
compile_convnet = true;
root = setup_path();
setup_convnet(use_gpu, compile_convnet);

% config
[config, net1] = coopnet_config(root);
config.use_gpu = use_gpu;
config.nIteration = 2;
config.num_syn = 2;

% prep
prefix = ['main/' num2str(img_size) '/'];
config = prep_images(config, [root 'data/' img_name '/' num2str(img_size) '/'], patch_size);
config = prep_dirs(config, prefix);

% run
learn_dual_net(config, net1);

end

function root = setup_path()
root = '../../../';
addpath([root 'src/training/coopnet']);
addpath([root 'src/training/matconvnet']);
addpath([root 'src/training/util']);
end