function res = vl_gan_gpu(net, x, dzdy, res, varargin)
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.mask = [];

opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
  dzdy = [];
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

% bottom up
for i = 1:n
    l = net.layers{i};
    res(i).time = tic;
    switch l.type
        case 'conv'
            res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
        case 'convt'
            res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                'crop', l.crop, 'upsample', l.upsample, ...
                'numGroups', l.numGroups, cudnn{:}) ;
        case 'normalize'
            res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
        case 'relu'
            if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
            res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
        case 'tanh'
            res(i+1).x = tanh(res(i).x);
        case 'bnorm'
            if isfield(l, 'weights')
                res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
            else
                res(i+1).x = vl_nnbnorm(res(i).x, l.filters, l.biases) ;
            end
        case 'sigmoid'
            res(i+1).x = vl_nnsigmoid(res(i).x) ;
        case 'custom'
            res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end
    
    forget = opts.conserveMemory ;
    forget = forget & (~doder || strcmp(l.type, 'relu')) ;
    forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
    if forget
        res(i).x = [] ;
    end
    if gpuMode & opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice) ;
    end
    res(i).time = toc(res(i).time) ;
end

% top down: 
if doder
    if isempty(dzdy)
        dzdy = gpuArray(ones(net.dydz_sz, 'single'));
    else
        dzdy = (dzdy - res(end).x);
    end
    if(~isempty(opts.mask))
        dzdy(opts.mask) = 0;
    end
    
    res(n+1).dzdx = dzdy;
    
    for i = n:-1:max(1, n-opts.backPropDepth+1)
        l = net.layers{i};
        res(i).backwardTime = tic;
        switch l.type
            case 'conv'
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                    vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                    res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride, ...
                    cudnn{:}) ;
            case 'convt'
                [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                    vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                    res(i+1).dzdx, ...
                    'crop', l.crop, 'upsample', l.upsample, ...
                    'numGroups', l.numGroups, cudnn{:}) ;
                
            case 'normalize'
                res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
            case 'relu'
                if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
                if ~isempty(res(i).x)
                    res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
                else
                    % if res(i).x is empty, it has been optimized away, so we use this
                    % hack (which works only for ReLU):
                    res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
                end
            case 'tanh'
                res(i).dzdx = (1 - res(i+1).x.^2) .* res(i+1).dzdx;
                
            case 'bnorm'
                if ~opts.accumulate
                    if isfield(l, 'weights')
                        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx) ;
                    else
                        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx) ;
                    end
                else
                    dzdw = cell(1,2) ;
                    if isfield(l, 'weights')
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx) ;
                    else
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx) ;
                    end
                    for j=1:2
                        res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                    end
                    clear dzdw ;
                end
            case 'sigmoid'
                res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
            case 'custom'
                res(i) = l.backward(l, res(i), res(i+1)) ;
        end
        if opts.conserveMemory
            res(i+1).dzdx = [] ;
        end
        if gpuMode & opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
end