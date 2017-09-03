function [alpha1,alpha2,barriers1,barriers2] = get_metastable_region(config,...
                                    des_net,gen_net,z1,z2,temps,alpha_init)
    % temperatures should be in increasing order
    
    barriers1 = zeros(length(temps),2);
    barriers2 = zeros(length(temps),2);
    alphas1 = zeros(1,length(temps));
	alphas2 = zeros(1,length(temps));
    
    for i = 1:length(temps)
        %disp(['****',num2str(i),'****']);
        config.attract_temp = temps(i);
        if i == 1 
            alpha1 = alpha_init; alpha2 = alpha_init;
        else
            alpha1 = alphas1(i-1)*change_factor^2; 
            alpha2 = alphas2(i-1)*change_factor^2; 
        end
        config.alpha = alpha1;
        [bar,a_bar,a_border] = find_metastable_border(config,des_net,gen_net,z1,z2);
        alphas1(i) = a_border;
        barriers1(i,:) = [bar,a_bar];
        %disp(alphas1(i));
        %disp(barriers1(i,2));
        %disp(barriers1(i,1));
        %disp('**')
        config.alpha = alpha2;
        [bar,a_bar,a_border] = find_metastable_border(config,des_net,gen_net,z2,z1);
        alphas2(i) = a_border;
        barriers2(i,:) = [bar,a_bar];
        %disp(alphas2(i));
        %disp(barriers2(i,2));
        %disp(barriers2(i,1));
    end

end