function [min_z,min_im,min_ind,min_energy,ens,im_path,z_path,accept_rate] = find_gen_min(config,des_net,gen_net,z)
    z_path = zeros([config.z_sz,config.min_sweeps+1]);
    im_path = zeros([config.im_sz,config.min_sweeps+1]);
   
    [en,im] = get_gen_energy(config,des_net,gen_net,z);
    ens_sz = 1; 
    MH_type = config.MH_type;
    if strcmp(MH_type,'RW')
        ens = flintmax*ones(1,config.min_sweeps+1);
    elseif strcmp(MH_type,'CW')
        ens_sz = length(z(:));
        ens = flintmax*ones(1,ens_sz*config.min_sweeps+1);
    else
        error('specify MH type in ADELM_config');
    end
    
    ens(1) = en;
    im_path(:,:,:,1) = im;
    z_path(:,:,:,1) = z;
    min_energy = en;
    min_ind = 1;
    
    accept_rate = 0;
    iter = 0;
    no_improve_counter = 0;
    while (iter < config.min_sweeps) && (no_improve_counter<config.min_no_improve)        
        iter = iter+1;
        %disp(iter);       
        %update z using low-temp MH proposal
        [z,new_im,en,accepted,ens_MH] = gen_MH_step(config,des_net,gen_net,z,en,config.min_temp);
        accept_rate = accept_rate + accepted;
        if ~isempty(new_im), im = new_im; end
        ens((2+(iter-1)*ens_sz):(1+(iter-1)*ens_sz+length(ens_MH))) = ens_MH;
        im_path(:,:,:,iter+1) = im;
        z_path(:,:,:,iter+1) = z;
           
        if en >= min_energy 
            no_improve_counter = no_improve_counter+1;
        else
            min_ind = iter+1;
            no_improve_counter = 0;
        end
        min_energy = min(en,min_energy);
    end
    
    min_im = im_path(:,:,:,min_ind);
    min_z = single(z_path(:,:,:,min_ind));
    ens = ens(ens<flintmax);
    im_path = im_path(:,:,:,1:(iter+1));
    accept_rate = accept_rate/iter;
end