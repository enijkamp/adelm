function [AD_out1,AD_out2] = gen_AD(config,des_net,gen_net,z1,z2,diff_ind)
    % "THE PERSISTENCE OF MEMORY"

    % diff_ind=1 or diff_ind=2 to diffuse from 1to2 or 2to1 only
    % leave empty to diffuse in both directions
    if nargin < 6 || isempty(diff_ind), diff_ind = [1,2]; end
    
    z1 = single(z1);
    z2 = single(z2);
    
    AD_out1 = [];
    AD_out2 = [];
    
    % Diffusion from im1 to im2
    if ismember(1,diff_ind)
        AD_out1 = AD_check(config,des_net,gen_net,z1,z2);
    end
    
    % Diffusion from im2 to im1
    if ismember(2,diff_ind)
        AD_out2 = AD_check(config,des_net,gen_net,z2,z1);
    end
end

function AD_out = AD_check(config,des_net,gen_net,z,z_target)
    % 0 indicates unsuccessful AD travel, 1 indicates successful AD travel 
    AD_out.mem = 0;
    
    %raw energy for each MH update along AD path
    ens_sz = 1; 
    if strcmp(config.MH_type,'RW')
        ens = flintmax*ones(1,config.max_AD_iter+1);
    elseif strcmp(config.MH_type,'CW')
        ens_sz = length(z(:));
        ens = flintmax*ones(1,ens_sz*config.max_AD_iter+1);
    else
        error('specify MH type in ADELM_config');
    end
    
    %image after each AD sweep
    im_path = zeros([config.im_sz,config.max_AD_iter+1]);
    % z path after each AD sweep
    z_path = zeros([config.z_sz,config.max_AD_iter+1]);
    % acceptance rate of MH proposals along AD path
    accept_rate = 0;
    %energy in magnetized landscape after each sweep
    a_ens = zeros(1,config.max_AD_iter+1);
    % distance between AD chain and target after each sweep
    dists = zeros(1,config.max_AD_iter+1);
    
    %initialize chain at z with target z_target
    curr_dist = norm(z(:)-z_target(:));
    min_dist = curr_dist;
    [en,im] = get_gen_energy(config,des_net,gen_net,z);
    
    ens(1) = en;
    a_ens(1) = en + config.alpha *norm(z(:)-z_target(:));
    dists(1) = curr_dist;
    im_path(:,:,:,1) =  im;
    z_path(:,:,:,1) = z;
    iter = 0;
    no_improve_counter = 0;
    % diffuse in altered landscape until until chain reaches target
    % OR cannot further approach target for AD_no_improve consecutive sweeps
    while iter < config.max_AD_iter && no_improve_counter < config.AD_no_improve ...
                                && curr_dist > config.dist_res
        iter = iter+1;
        
        % single AD sweep of chain
        [z,new_im,en,accepted,ens_MH] = gen_MH_step(config,des_net,gen_net,z,en,config.AD_temp,config.alpha,z_target);
        
        % update AD record
        if ~isempty(new_im), im = new_im; end
        accept_rate = accept_rate+accepted;  
        ens((2+(iter-1)*ens_sz):(1+iter*ens_sz)) = ens_MH;
        a_ens(iter+1) = en + config.alpha*norm(z(:)-z_target(:));
        im_path(:,:,:,iter+1) = im;
        z_path(:,:,:,iter+1) = z;
        curr_dist = norm(z(:) - z_target(:));
        dists(iter+1) = curr_dist;
        %disp(curr_dist);
        
        %check if chain has improved on previous min distance to target
        if curr_dist >= min_dist 
            no_improve_counter = no_improve_counter+1;
        else
            no_improve_counter = 0;
        end
        min_dist = min(curr_dist,min_dist);
    end
    
    if norm(z(:)-z_target(:)) <= config.dist_res, AD_out.mem = 1; end 
    AD_out.ens = ens(ens<flintmax);
    AD_out.a_ens = a_ens(1:(iter+1));
    AD_out.im_path = im_path(:,:,:,1:(iter+1));
    AD_out.z_path = z_path(:,:,:,1:(iter+1));
    AD_out.dists = dists(1:(iter+1));
    AD_out.accept_rate = accept_rate/iter;
end