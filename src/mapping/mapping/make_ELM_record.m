function ELM = make_ELM_record(config,des_net,gen_net,min_z)    
    ELM.config = config;
    
    %info about global basin reps
    ELM.min_z = min_z;
    for i = 1:size(min_z,4)
        [en_i,min_i] = get_gen_energy(config,des_net,gen_net,min_z(:,:,:,i));
        ELM.min_ims(:,:,:,i) = min_i;
        ELM.min_ens(i) = en_i;
    end
    ELM.min_IDs = 1:size(min_z,4);
    
    %info about ELM mapping chain
    ELM.new_chain = 0;  
    ELM.z_path = zeros([config.z_sz,0]);
    ELM.im_path = zeros([config.im_sz,0]);
    ELM.min_im_path = zeros([config.im_sz,0]);
    ELM.min_z_path = zeros([config.z_sz,0]);
    ELM.min_ID_path= [];
    ELM.en_path = [];
    ELM.min_en_path = [];
    ELM.mins_discarded = 0;
end