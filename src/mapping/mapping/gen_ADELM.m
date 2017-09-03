function ELM = gen_ADELM(ELM)

    rng(123);

    addpath(fullfile('../viz'));
    addpath(fullfile('../tools'));
    addpath(fullfile('../sampling'));
    addpath(fullfile('../postmapping'));
    addpath(fullfile('../mapping'));
    addpath(fullfile('../config'));
    
    current_dir = pwd();
    cd(fullfile('../../matconvnet-1.0-beta16/', 'matlab'));
    vl_setupnn();
    %vl_compilenn();
    cd(current_dir);
  
    if nargin < 1 || isempty(ELM)
        %read in config parameters
        config = gen_ADELM_config(1);
        des_net = config.des_net;
        gen_net = config.gen_net;
        
        %start chain
        z = randn(config.z_sz,'single');
        [en,im] = get_gen_energy(config,des_net,gen_net,z);
        [min_z,min_im,~,min_en] = find_gen_min(config,des_net,gen_net,z);

        %make new record
        ELM = make_ELM_record(config,des_net,gen_net,min_z);
        ELM = update_ELM_record(ELM,z,im,min_z,min_im,en,min_en,1); 
    else
        % load config from old ELM
        config = ELM.config;
        des_net = config.des_net;
        gen_net = config.gen_net;
        ELM.new_chain(end+1) = length(ELM.min_ID_path)+1;
    end
    
    viz_min_ims(ELM.min_ims,config.im_folder);
    for rep = 1:config.nsteps    
        fprintf('\n');
        fprintf('ELM Step %d of %d\n',rep,config.nsteps);
        fprintf('----\n');
        % find new min and classify in each ELM step
        ELM = gen_ADELM_step(config,des_net,gen_net,ELM);
   
        %save results
        viz_min_ims(ELM.min_ims,config.im_folder,1);
        save([config.ELM_folder,config.ELM_str,'ELM.mat'],'ELM');
    end  
end