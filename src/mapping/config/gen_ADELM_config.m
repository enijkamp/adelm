function config = gen_ADELM_config(load_nets)
    % .mat files with descriptor and generator nets (load_nets=1 to load)
    config.ELM_str = 'ivy2/512_2/';
    config.net_file = 'nets.mat';
    % digits/ , ivy/224/
    
    
    %MH parameters
    config.refsig = 1;
    config.MH_eps = 0.05; %step size
    config.MH_type = 'RW'; % 'RW' (random walk) or 'CW' (component-wise/gibbs)

    % parameters for ADELM
    config.nsteps = 70; % number of ELM iterations
    config.num_mins = 200; % max number of basins on record
    config.AD_heuristic = '1D_bar'; % 1D linear interpolation '1D_bar'
                                       % or Euclidean dist 'dist'
                                      % determines AD order

    %local min search
    config.min_temp = .1; %temperature for min search
    config.min_sweeps = 5000; % max number of sweeps during min search
    config.min_no_improve = 30; % consecutive failed iters to stop search

    %attraction diffusion
    config.AD_temp = 20; % AD temperature parameter
    config.alpha = 1400; % AD magnetization strength
    %4700
    config.max_AD_iter = 5000;  % max iters for AD trial
    config.AD_no_improve = 40; % consecutive iters to stop search
    config.dist_res = .35; % distance from target for successful AD search
    config.max_AD_checks = 15; % number of minima for AD trials
    config.AD_reps = 1; % number of AD attempts for each min 
    config.AD_quota = 1; % number of successful trials needed for membership
    config.update_min_states = 1; % change basin reps (1) or not (0)
    
    % parameters for barrier estimation
    config.bar_temp = 20;
    config.bar_alpha = 1.4; 
    config.bar_factor = 1.05; %mutiply mag. force by this during bar search
    config.bar_checks = 3; % number of AD trials during bar search
        
    % data location
    config.data_path = '../../data/';
    % location of Co-op Nets
    config.net_path = '../../nets/';
    % folder for ELM results
    config.ELM_folder = '../../maps/';
    % folder for images in generator space
    config.im_folder = '../../ims/';
    % folder for ELM Trees
    config.tree_folder = '../../trees/';

    % create results directory
    if ~exist('../../maps/', 'dir')
        mkdir('../../maps/')
    end

    if ~exist('../../ims/', 'dir')
        mkdir('../../ims/')
    end
    
    if ~exist('../../trees/', 'dir')
        mkdir('../../trees/')
    end
    
    if nargin >= 1 && load_nets == 1
        if strcmp(config.ELM_str,'digits/')
            net_wkspc = load([config.net_path,config.ELM_str,config.net_file]);
            config.des_net = net_wkspc.nets.des_net;
            config.gen_net = net_wkspc.nets.gen_net;
            config.mean_im = config.des_net.mean_im;
            config.z_sz = [1,1,8];
            config.im_sz = [64,64,1];
        end
        
        if strcmp(config.ELM_str(1:3),'ivy')
            net_wkspc = load([config.net_path,config.ELM_str,config.net_file]);
            config.des_net = net_wkspc.net1;
            config.gen_net = net_wkspc.net2;
            config.mean_im = config.des_net.mean_im;
            config.z_sz = [1,1,15];
            config.im_sz = [32,32,3];
        end
    end
    
    %config.z_sz = [1, 1, size(gen_net.layers{1}.weights{1}, 4)];
    %config.z_sz = [1,1,8];
    %config.im_sz = [64,64,1];
end