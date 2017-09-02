function [gen_im,z] = get_gen_im(gen_net,z,z_sz)
    if isempty(z), z = randn(z_sz,'single'); end
    syn_mat = vl_gan_cpu(gen_net,z);
    gen_im = floor((syn_mat(end).x+1)*128);
end