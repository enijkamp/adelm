function [] = exp_texture()

rng(123);

% config
img_name = 'ivy2';
img_sizes = 256; %2.^(6:13);
patch_size = 32;
patch_n = 300;

% setup
use_gpu = 0;
compile_convnet = 1;
create_patches = 0;

root = setup_path();
setup_convnet(use_gpu, compile_convnet);

% sample texture patches
if create_patches
    sample_patches([root 'data/' img_name '/original/texture.png'], img_name, img_sizes, patch_size, patch_n);
end

% train coop nets per scale
for img_size = img_sizes
    % config
    prefix = [img_name '/' num2str(img_size) '/'];
    [config, net1] = coopnet_config(root);
    config = prep_dirs(config, prefix);
    config = prep_images(config, [root 'data/' prefix], patch_size);
    config.use_gpu = use_gpu;
    
    % train
    learn_dual_net(config, net1);
end

end

function [config] = prep_images(config, patch_path, patch_size)
[mean_im, imdb] = load_images(patch_path, patch_size);
config.mean_im = mean_im;
config.imdb = imdb;
end

function [config] = prep_dirs(config, prefix)
config.trained_folder = [config.trained_folder prefix];
config.gen_im_folder = [config.gen_im_folder prefix];
config.syn_im_folder = [config.syn_im_folder prefix];
if ~exist(config.trained_folder,'dir') mkdir(config.trained_folder); end
if ~exist(config.gen_im_folder,'dir') mkdir(config.gen_im_folder); end
if ~exist(config.syn_im_folder,'dir') mkdir(config.syn_im_folder); end
end

function sample_patches(img_path, texture_name, img_sizes, pat_size, pat_n)
img = imread(img_path);
for sz = img_sizes
    img_res = imresize(img, [sz, sz]);
    
    pat_dir = ['../data/' texture_name '/' num2str(sz) '/'];
    if ~exist(pat_dir, 'dir')
        mkdir(pat_dir);
    end
    
    for i = 1:pat_n
        pat = img_res(randi(sz-pat_size+1)+(0:pat_size-1),randi(sz-pat_size+1)+(0:pat_size-1),:);
        imwrite(pat, [pat_dir sprintf('%03d.png',i) ]);
    end
end
end

function [mean_im, imdb] = load_images(img_path, img_size)
files = dir([img_path, '*.png']);
imdb = zeros(img_size, img_size,3,length(files));
for i = 1:length(files)
    imdb(:,:,:,i) = imread([img_path,files(i).name]);
end
mean_im = single(sum(imdb,4)/size(imdb,4));
imdb = single(imdb - repmat(mean_im,1,1,1,size(imdb,4)));
end
