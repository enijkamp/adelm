function [z,im,en,accepted,ens] = gen_MH_step(config,des_net,gen_net,z,en,temp,alpha,z_target)
    if nargin < 5 || isempty(en), en = get_gen_energy(config,des_net,gen_net,z); end
    if nargin < 6 || isempty(temp), temp = 1; end
    if nargin < 7 || isempty(z_target), alpha = 0; z_target = zeros(size(z)); end
    
    MH_type = config.MH_type;
    if strcmp(MH_type,'RW')
        %sample using random-walk M-H proposal (one RW step)
        [z,im,en,accepted] = RW_MH_step(config,des_net,gen_net,z,en,temp,alpha,z_target);
        ens = en;
    elseif strcmp(MH_type,'CW')
        %sample using component-wise (gibbs) M-H proposal (one sweep)
        [z,im,en,accepted,ens] = CW_MH_step(config,des_net,gen_net,z,en,temp,alpha,z_target);
    end
end

function [z,im,en,accepted] = RW_MH_step(config,des_net,gen_net,z,en,temp,alpha,z_target)
    accepted = 0;
    im = [];
    
    zstar = z + config.MH_eps*randn(size(z));
    [prop_en,prop_im] = get_gen_energy(config,des_net,gen_net,zstar);
    
    %update according to MH acceptance with AD penalty
    %normal diffusion when alpha = 0
    prob = exp(-(prop_en/temp+alpha*norm(zstar(:)-z_target(:))) + ...
                (en/temp+alpha*norm(z(:)-z_target(:))));
            
    if rand < prob, z = zstar; en = prop_en; accepted = 1; im = prop_im; end
end

function [z,im,en,accepted,ens] = CW_MH_step(config,des_net,gen_net,z,en,temp,alpha,z_target)
    accepted = 0;
    im = [];
    ens = zeros(1,length(z(:)));
    
    gibbs_order = randperm(length(z(:)));
    for j = 1:length(z(:))
        i = gibbs_order(j);
        zstar = z;
        zstar(i) = z(i) + config.MH_eps*randn(1,'single');
        [prop_en,prop_im] = get_gen_energy(config,des_net,gen_net,zstar);
        %update according to MH acceptance with AD penalty
        %normal diffusion when alpha = 0
        prob = exp(-(prop_en/temp+alpha*norm(zstar(:)-z_target(:)) + ...
                    (en/temp+alpha*norm(z(:)-z_target(:)))));

        if rand < prob
            z(i) = zstar(i); 
            en = prop_en; 
            accepted = accepted + 1; 
            im = prop_im;
        end
        ens(j) = en;
    end
    accepted = accepted/length(z(:));
end