function [array] = toGpuArray(array, use_gpu)
if use_gpu
    array = gpuArray(array);
end
end