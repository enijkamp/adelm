function [] = exp_texture_grid()

rng(123);

% config
img_name = 'ivy2';
%img_sizes = 2.^(6:13);
img_sizes = 2^8;
patch_size = 32;
patch_n = 300;

% setup
use_gpu = 1;
compile_convnet = 0;
create_patches = 0;

% setup convnet
setup_convnet(use_gpu, compile_convnet);

% sample texture patches
if create_patches
    sample_patches(['../data/' img_name '/original/texture.png'], img_name, img_sizes, patch_size, patch_n);
end

%% (1)
% % grid config
% grid.num_syn = [1];
% grid.Delta = [0.03, 0.3];
% grid.Gamma = [0.000005, 0.00005, 0.0005];
% grid.refsig = [1]; 
% grid.T = [15];  
% 
% grid.Delta2 = [0.3];
% grid.Gamma2 = [0.000035, 0.00005, 0.0005];
% grid.refsig2 = [1];
% grid.s = [0.3];
% grid.real_ref = [1];
% grid.cap2 = [8];

% num_syn   delta     gamma     refsig   T         delta2     gamma2    refsig2   s        real_ref    cap2

% 1.0000    0.0300    0.0000    1.0000   15.0000    0.3000    0.0000    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.0300    0.0000    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.0300    0.0000    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.0300    0.0001    1.0000   15.0000    0.3000    0.0000    1.0000    0.3000    1.0000    8.0000 % bad, bumpy

% 1.0000    0.0300    0.0001    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, bumpy
% 1.0000    0.0300    0.0001    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % ok-ish, bumpy
% 1.0000    0.0300    0.0005    1.0000   15.0000    0.3000    0.0000    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.0300    0.0005    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % ok-ish, bumpy

% 1.0000    0.0300    0.0005    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % ok-ish, bumpy
% 1.0000    0.3000    0.0000    1.0000   15.0000    0.3000    0.0000    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.3000    0.0000    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.3000    0.0000    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % bad, uniform

% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0000    1.0000    0.3000    1.0000    8.0000 % ok-ish, bumpy
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % good, bumpy
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % good, bumpy
% 1.0000    0.3000    0.0005    1.0000   15.0000    0.3000    0.0000    1.0000    0.3000    1.0000    8.0000 % bad, uniform

% 1.0000    0.3000    0.0005    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, bumpy
% 1.0000    0.3000    0.0005    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % good, bumpy

%           0.3000    0.0005                                  0.0005

%% (2)
% % grid config
% grid.num_syn = [1];
% grid.Delta = [0.3];
% grid.Gamma = [0.00005, 0.0001, 0.0005];
% grid.refsig = [1]; 
% grid.T = [15, 40];  
% 
% grid.Delta2 = [0.3];
% grid.Gamma2 = [0.00005, 0.0001, 0.0005];
% grid.refsig2 = [1];
% grid.s = [0.3];
% grid.real_ref = [1];
% grid.cap2 = [8];

% num_syn   delta     gamma     refsig   T         delta2     gamma2    refsig2   s        real_ref    cap2

% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % good
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % good, less
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % good, bumpy
% 1.0000    0.3000    0.0001    1.0000   40.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % ok-ish
% 1.0000    0.3000    0.0001    1.0000   40.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % ok-ish

% 1.0000    0.3000    0.0001    1.0000   40.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % good
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % ok-ish, bumpy
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, fade
% 1.0000    0.3000    0.0001    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % bad, uenven
% 1.0000    0.3000    0.0001    1.0000   40.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, fade

% 1.0000    0.3000    0.0001    1.0000   40.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, uenven
% 1.0000    0.3000    0.0001    1.0000   40.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % bad, uenven
% 1.0000    0.3000    0.0005    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, almost uniform
% 1.0000    0.3000    0.0005    1.0000   15.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, bumpy
% 1.0000    0.3000    0.0005    1.0000   15.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % bad, bumpy

% 1.0000    0.3000    0.0005    1.0000   40.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, uniform
% 1.0000    0.3000    0.0005    1.0000   40.0000    0.3000    0.0001    1.0000    0.3000    1.0000    8.0000 % bad, bumpy
% 1.0000    0.3000    0.0005    1.0000   40.0000    0.3000    0.0005    1.0000    0.3000    1.0000    8.0000 % bad, bumpy


configs = cartesian(grid.num_syn, grid.Delta, grid.Gamma, grid.refsig, grid.T, ...
    grid.Delta2, grid.Gamma2, grid.refsig2, grid.s, grid.real_ref, grid.cap2);

% train coop nets per scale
for img_size = img_sizes
    for i = 1:size(configs,1)
        % prep
        prefix = [img_name '/256_grid_' num2str(i) '/'];
        [config, net1] = train_coop_config();
        config = prep_images(config, ['../data/' img_name '/256/'], patch_size);
        config = prep_dirs(config, prefix);
        config.use_gpu = use_gpu;
        
        % sampling parameters
        config.num_syn = configs(i, 1);
        % descriptor net1 parameters
        config.Delta = configs(i, 2);
        config.Gamma = configs(i, 3);
        config.refsig = configs(i, 4);
        config.T = configs(i, 5);
        % generator net2 parameters
        config.Delta2 = configs(i, 6);
        config.Gamma2 = configs(i, 7);
        config.refsig2 = configs(i, 8);
        config.s = configs(i, 9);
        config.real_ref = configs(i, 10);
        config.cap2 = configs(i, 11);
        
        % learn
        learn_dual_net(config, net1);
    end
end

end

function C = cartesian(varargin)
args = varargin;
n = nargin;
[F{1:n}] = ndgrid(args{:});
for i=n:-1:1
    G(:,i) = F{i}(:);
end
C = unique(G , 'rows');
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