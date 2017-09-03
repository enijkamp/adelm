function ELM = update_ELM_record(ELM,z,im,min_z,min_im,en,min_en,min_ID)
    if ELM.new_chain(1) == 0, ELM.new_chain = 1; end
    ELM.z_path(:,:,:,end+1) = z;
    ELM.min_z_path(:,:,:,end+1) = min_z;
    ELM.im_path(:,:,:,end+1) = im;
    ELM.min_im_path(:,:,:,end+1) = min_im;
    ELM.min_ID_path(end+1) = min_ID;
    ELM.en_path(end+1) = en;
    ELM.min_en_path(end+1) = min_en;
end