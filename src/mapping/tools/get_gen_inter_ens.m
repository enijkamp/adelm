function [inter_ens,inter_ims] = get_gen_inter_ens(config,des_net,gen_net,z1,z2,num_points)
    if nargin< 6 || isempty(num_points), num_points = 10; end
    
    % get 1D linear interpolation energy landscape between two points
    num_points = num_points-1;
    inter_ims = zeros([config.im_sz,num_points+1]);
    inter_ens = zeros(1,num_points+1);
    for i = 1:(num_points+1)
        z = single(((i-1)/num_points)*z2 + ((num_points-i+1)/num_points)*z1);
        [en,im] = get_gen_energy(config,des_net,gen_net,z);
        inter_ims(:,:,:,i) = im;
        inter_ens(i) = en;
    end
end