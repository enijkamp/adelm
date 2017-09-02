function net = add_bottom_filters(net, layer, config)

%parameters for layer initialization
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;

if ismember(1, layer)
    %% layer 1
    layer_name = '1';
    num_in = 3;
    num_out = 200; % number of filters in the first layer
    filter_sz = 11; % the square filter with filter size 15x15 in the 1st layer (7)
    stride = 2; % sub-sampling size for each filter, 1 means in spatial space, each filter is put every two pixels. 
    pad_sz = floor(filter_sz/2); % pad size for image. For example, image is 224x224, and we add two pixels (padded image is 228x228) to handle boundary conditions.
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
end

if ismember(2, layer)
    %% layer 2
    layer_name = '2';
    num_in = 200;
    num_out = 100; % number of filters in the first layer
    filter_sz = 7; % the square filter with filter size 15x15 in the 1st layer
    stride = 2; % sub-sampling size for each filter, 1 means in spatial space, each filter is put every two pixels. 
    pad_sz = floor(filter_sz/2); % pad size for image. For example, image is 224x224, and we add two pixels (padded image is 228x228) to handle boundary conditions.
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
end

if ismember(3, layer)
    %% layer 2
    layer_name = '3';
    num_in = 100;
    num_out = 1; % number of filters in the first layer
    filter_sz = 8; % the square filter with filter size 15x15 in the 1st layer
    stride = 1; % sub-sampling size for each filter, 1 means in spatial space, each filter is put every two pixels. 
    pad_sz = 0; % pad size for image. For example, image is 224x224, and we add two pixels (padded image is 228x228) to handle boundary conditions.
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
end