function [barrier_mat,alpha_mat,sym_mat] = get_barrier_mat(config,des_net,...
            gen_net,min_z)
    %NOTE: gives the raw energy at the barriers between nodes, NOT
    % the difference in energy between min state and energy at barrier

    num_mins = size(min_z,4);
    
    if num_mins == 1
        barrier_mat = get_gen_energy(config,des_net,gen_net,single(min_z(:,:,:,1)));
        sym_mat = barrier_mat;
        alpha_mat = 0;
        return; 
    end
    
    barrier_mat = flintmax*ones(num_mins,num_mins);
    sym_mat = barrier_mat;
    alpha_mat = zeros(num_mins,num_mins);
    
    for i = 1:(num_mins-1)
        z_i = single(min_z(:,:,:,i));
        for j = (i+1):num_mins
            %disp('****');
            z_j = single(min_z(:,:,:,j));
            [alpha1,alpha2,bars1,bars2] = get_metastable_region(config,...
                des_net,gen_net,z_i,z_j,config.bar_temp,config.bar_alpha);
            alpha_mat(i,j) = alpha1;
            alpha_mat(j,i) = alpha2;
            barrier_mat(i,j) = bars1(1,1);
            barrier_mat(j,i) = bars2(1,1);
        end
    end
    
    for i = 1:num_mins
        for j = i:num_mins
            if i == j
                barrier_mat(i,i) = get_gen_energy(config,des_net,gen_net,...
                                        single(min_z(:,:,:,i))); 
                sym_mat(i,i) = barrier_mat(i,i);
            else
                sym_mat(i,j) = min([barrier_mat(i,j),barrier_mat(j,i)]);
                sym_mat(j,i) = sym_mat(i,j);
            end
        end
    end   
end