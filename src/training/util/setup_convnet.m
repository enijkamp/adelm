function [] = setup_convnet(use_gpu, do_compile)

current_dir = pwd();

if use_gpu
    % gpu
    cd(fullfile('../../../libs/matconvnet-1.0-beta16-gpu/', 'matlab'));
    cuda_root = ''; 
    
    vl_setupnn();
    
    if ispc
        cuda_method = 'nvcc';
    else
        cuda_method = 'mex';
    end
    
    if do_compile
        if isempty(cuda_root)
            vl_compilenn('EnableGPU', true, 'CudaMethod', 'nvcc');
        else
            vl_compilenn('EnableGPU', true, 'CudaRoot', cuda_root, 'CudaMethod', cuda_method);
        end
    end
else
    % cpu
    cd(fullfile('../../../libs/matconvnet-1.0-beta16/', 'matlab'));
    vl_setupnn();
    if do_compile
        vl_compilenn();
    end
end

cd(current_dir);

end
