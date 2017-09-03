function res = vl_gan(net, x, dzdy, res, varargin)

gpuMode = isa(x, 'gpuArray');
if gpuMode
    if (nargin <= 2)
        res = vl_gan_gpu(net, x);
    elseif (nargin <= 3)
        res = vl_gan_gpu(net, x, dzdy);
    elseif (nargin <= 4)
        res = vl_gan_gpu(net, x, dzdy, res);
    else
        res = vl_gan_gpu(net, x, dzdy, res, varargin{:});
    end
else
    if (nargin <= 2)
        res = vl_gan_cpu(net, x);
    elseif (nargin <= 3)
        res = vl_gan_cpu(net, x, dzdy);
    elseif (nargin <= 4)
        res = vl_gan_cpu(net, x, dzdy, res);
    else
        res = vl_gan_cpu(net, x, dzdy, res, varargin{:});
    end
end

end