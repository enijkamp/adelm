function ELM = gen_ADELM_step(config,des_net,gen_net,ELM)    
    %generate z from latent normal distribution
    z = randn(config.z_sz,'single');
    [en,im] = get_gen_energy(config,des_net,gen_net,z);
    imwrite(im/256,[config.im_folder,'gen_im.png']);
    
    % find local min of z in generator space
    disp('Finding local min');
    min_search_time = tic;
    [min_z,min_im,~,min_en] = find_gen_min(config,des_net,gen_net,z);
    min_search_time = toc(min_search_time);
    fprintf('%4.2f seconds \n',min_search_time);
    fprintf('----\n');
    
    imwrite(min_im/256,[config.im_folder,'min_im.png']);
   
    % determine if this local min can be identified with previous min
    disp('Checking minimum membership');
    AD_time = tic;
    [ELM,min_index] = check_membership(config,des_net,gen_net,ELM,min_z,...
                                            min_en,min_im);
    AD_time = toc(AD_time);
    fprintf('%4.2f seconds \n',AD_time);
    
    %include the new minimum in the mapping record
    if min_index <= config.num_mins
        ELM = update_ELM_record(ELM,z,im,min_z,min_im,en,min_en,...
                ELM.min_IDs(min_index)); 
    end
    fprintf('----\n');
end

function AD_order = get_AD_order(config,des_net,gen_net,min_z_mat,min_z,...
                                                        min_ens,min_en)
    if strcmp(config.AD_heuristic,'1D_bar')
        barrier1D = flintmax*ones(1,size(min_z_mat,4));
        for i = 1:length(barrier1D)          
            barrier1D(i) = max(get_gen_inter_ens(config,des_net,gen_net, ...
                single(min_z_mat(:,:,:,i)),min_z)); % ...
                  %- max([min_en,min_ens(i)]);
        end
        [~,AD_order] = sort(barrier1D);
    elseif strcmp(config.AD_heuristic,'dist')
        dists = flintmax*ones(1,size(min_z_mat,4));
        for i = 1:length(dists)
            dists(i) = norm(min_z(:)-reshape(min_z_mat(:,:,:,i),[],1));
        end
        [~,AD_order] = sort(dists);
    else
        error('need correct type for config.AD_heuristic');
    end
end

function [ELM,min_index] = check_membership(config,des_net,gen_net,ELM,...
                                                min_z,min_en,min_im)
    min_index = 0;
    % use heuristic (1D barrier or distance) to find order for AD
    AD_order = get_AD_order(config,des_net,gen_net,ELM.min_z,min_z,...
                                                ELM.min_ens,min_en);  
    
    %check membership according to the energy of the n closest local
    %minima, according to heuristic (config.max_AD_checks)
    AD_mem = zeros(min(config.max_AD_checks,length(AD_order)),2);
    AD_bars = flintmax*ones(1,min(config.max_AD_checks,length(AD_order)));
    for rep = 1:config.AD_reps
        for i = 1:min(config.max_AD_checks,length(AD_order))           
            AD_index = AD_order(i);
            %disp([m_rep,sort_index]);
            % AD diffusion between new and previously found minima
            [AD_out1,AD_out2]=gen_AD(config,des_net,gen_net,min_z,...
                        ELM.min_z(:,:,:,AD_index));
            AD_mem(i,:) = [AD_out1.mem,AD_out2.mem];
            AD_bars(i) = min([max(AD_out1.ens),max(AD_out2.ens)]);
        end
        
        %check if successful diffusion quota for membership is reached
        if max(sum(AD_mem)) >= config.AD_quota
            mem_inds = find(sum(AD_mem)==max(sum(AD_mem)));
            min_index = AD_order(mem_inds(find(AD_bars(mem_inds)==...
                                    min(AD_bars(mem_inds)),1,'first')));            
            fprintf('min sorted to basin %d\n',min_index);
            % if the new min has lower energy than the previous
            % basin rep, it becomes the new basin rep
            if min_en < ELM.min_ens(min_index) && config.update_min_states==1
                ELM.min_ims(:,:,:,min_index) = min_im;
                ELM.min_z(:,:,:,min_index) = min_z;
                ELM.min_ens(min_index) = min_en;
            end
            break;
        end
    end

    %if no AD chain is successful, start a new min group
    if min_index == 0
        if length(ELM.min_IDs) < config.num_mins
            min_index = length(ELM.min_IDs)+1;
            fprintf('new min found (ID %d)\n',min_index);
            ELM.min_ens(min_index) = min_en;
            ELM.min_ims(:,:,:,min_index) = min_im;
            ELM.min_z(:,:,:,min_index) = min_z;
            ELM.min_IDs(min_index) = max(ELM.min_IDs)+1;
        elseif min_en < max(ELM.min_ens)
            min_index = find(ELM.min_ens==max(ELM.min_ens),1,'first');
            fprintf('*New min found (ID %d)*\n',min_index);
            ELM.min_ens(min_index) = prop_min_energy;
            ELM.min_ims(:,:,:,min_index) = min_im;
            ELM.min_z(:,:,:,min_index) = min_z;
            ELM.min_IDs(min_index) = max(ELM.min_IDs)+1;
        else
            fprintf('min discarded \n');
            ELM.mins_discarded = ELM.mins_discarded+1;
            min_index = config.num_mins+1;
        end
    end
end