function [ord,inds,ens]= viz_basin_ims(ELM,index,im_folder,config,des_net,keep)
    if isempty(im_folder), im_folder = config.im_folder; end
    if nargin<6 || isempty(keep)
        delete([im_folder,'*.png']);
    end
    inds = find(ismember(ELM.min_ID_path,index));
    ens = zeros(1,length(inds));
    if nargin > 3
        for i = 1:length(inds)
            ens(i) = get_im_energy(config,des_net,ELM.min_im_path(:,:,:,inds(i)));
        end
        [~,ord]=sort(ens);
    else
        ord = 1:length(inds);
    end
    disp(length(ord));
    for i = 1:length(index)
        imwrite(ELM.min_ims(:,:,:,index(i))/256,[im_folder,...
                'basin',num2str(index(i)),'-min.png']);
    end
        
    for i = 1:length(ord)
        imwrite(ELM.min_im_path(:,:,:,inds(ord(i)))/256,[im_folder,...
            'basin',num2str(ELM.min_ID_path(inds(ord(i)))),'-',num2str(i),'.png']);
    end
end