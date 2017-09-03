function [config,net_cpu] = coopnet_config(root)

% num epochs
config.nIteration = 200;
config.batchSize = 32;

% sampling parameters
config.num_syn = 32;

% descriptor net1 parameters
config.Delta = 0.3;
config.Gamma = 0.00005 * logspace(-2, -3, config.nIteration)*100; % learning rate
config.Gamma_mul = logspace(-2, -3, config.nIteration);
config.refsig = 1; % standard deviation for reference model q(I/sigma^2).
config.T = 15;

% generator net2 parameters
config.Delta2 = 0.3;
config.Gamma2 =  0.00005 * logspace(-2, -3, config.nIteration)*100; % learning rate
config.refsig2 = 1;
config.s = 0.3;
config.real_ref = 1;
config.cap2 = 8;
config.infer_z = true;

% how many layers to learn
config.layer_to_learn = 1;

% image size
config.im_size = 32;

% batch function
fn = @(imdb,batch) getBatch(imdb,batch);
config.getBatch = fn;

% image path: where the dataset locates
config.inPath = [root 'data/'];

% model path: where the deep learning model locates
config.model_path = [root 'data/model/'];
config.model_name = 'imagenet-vgg-verydeep-16.mat';

% set up empty net
net_cpu = load([config.model_path, config.model_name]);
net_cpu = net_cpu.net;

% name folders for results
config.syn_im_folder = [root 'output/ims_syn/'];
config.gen_im_folder = [root 'output/ims_gen/'];
config.trained_folder = [root 'output/nets/'];

end

function im = getBatch(imdb, batch)
im = imdb(:,:,:,batch);
end