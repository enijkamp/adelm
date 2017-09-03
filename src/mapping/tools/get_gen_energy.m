function [energy,im] = get_gen_energy(config,des_net,gen_net,z)
    if ~isa(z,'single'), z = single(z); end
    im = get_gen_im(gen_net,z);
    energy = get_im_energy(config,des_net,im);
end