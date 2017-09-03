function [] = experiment_mnist()

rng(123);

% setup
compile_convnet = 1;
use_gpu = 0;

% compile convnet
root = setup_path();
setup_convnet(use_gpu, compile_convnet);

% train coop nets
for img_size = img_sizes
    % config
    prefix = 'mnist/';
    [config, net1] = coopnet_config(root);
    config.digits = 0:9; %digits to be used in the model
    config.set = 'test'; %train, test, both
    
    [imdb,im_mat, im_labs, mean_im,getBatch] = read_MNIST(config);
    config.imdb = single(imdb);
    config.imdb_mean = imresize(mean_im,[config.im_size,config.im_size]);
    config.im_mat = im_mat;
    config.im_labs = im_labs;
    %config.mean_im = single(zeros(config.im_size));
    config.mean_im = config.imdb_mean;
    config.imdb = config.imdb - repmat(config.mean_im,[1,1,1,size(config.imdb,4)]);
    
    config.use_gpu = use_gpu;
    
    config.trained_folder = [config.trained_folder prefix];
    config.gen_im_folder = [config.gen_im_folder prefix];
    config.syn_im_folder = [config.syn_im_folder prefix];
    
    % dirs
    if ~exist(config.trained_folder,'dir') mkdir(config.trained_folder); end
    if ~exist(config.gen_im_folder,'dir') mkdir(config.gen_im_folder); end
    if ~exist(config.syn_im_folder,'dir') mkdir(config.syn_im_folder); end
    
    % train
    learn_dual_net(config, net1);
end

end
