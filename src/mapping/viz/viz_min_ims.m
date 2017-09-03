function viz_min_ims(min_ims,im_folder,keep)
    if nargin < 3 || isempty(keep)
        delete([im_folder,'*.png']);
    end
    
    for i = 1:size(min_ims,4)
        imwrite(min_ims(:,:,:,i)/256,[im_folder,'min_im',num2str(i),'.png']);
    end
end