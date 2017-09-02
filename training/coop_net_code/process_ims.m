function process_ims(process_str,save_str,factor,patch_size,num_patch)
    im = imread(process_str);
    im = imresize(im,factor);
    for i = 1:num_patch
        start_pix = randi(min(size(im,1),size(im,2))-patch_size+1);
        imwrite(im(start_pix:(start_pix+patch_size-1),...
            start_pix:(start_pix+patch_size-1),:),[save_str,num2str(i),'.png']);
    end
end