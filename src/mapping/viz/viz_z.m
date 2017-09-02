function im = viz_z(gen_net,z,z_sz)
    if ~isempty(z), im = get_gen_im(gen_net,z);
    else, im = get_gen_im(gen_net,[],z_sz); end
    figure(1);
    imshow(im/256);
    figure(1);
end