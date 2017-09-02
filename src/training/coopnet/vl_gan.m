function res = vl_gan(net, x, dzdy, res, varargin)

gpuMode = isa(x, 'gpuArray');
if gpuMode
    res = vl_gan_gpu(net, x, dzdy, res, varargin);
else
    res = vl_gan_cpu(net, x, dzdy, res, varargin);
end

end