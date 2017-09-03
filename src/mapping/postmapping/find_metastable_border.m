function [barrier,a_bar,a_border] = find_metastable_border(config,des_net,gen_net,z,z_target)
    
    z = single(z);
    z_target = single(z_target);
    
    mem = 0;
    count = 0;
    barrier = flintmax;
    a_bar = config.alpha;
    while mem == 0 && count < config.bar_checks
        AD_out = gen_AD(config,des_net,gen_net,z,z_target,1);
        bar_est = flintmax;
        if AD_out.mem == 1, bar_est = max(AD_out.ens); end
        bar_out = max([bar_est, ...
            get_gen_energy(config,des_net,gen_net,z), ...
                get_gen_energy(config,des_net,gen_net,z_target)]);
        if bar_out < barrier, barrier = bar_out; a_bar=config.alpha; end
        mem = mem + AD_out.mem;
        count = count+1;
    end

    sgn = 1*(mem>0);
    mem = sgn;
    while mem == sgn
        if sgn == 0; config.alpha = config.alpha*config.bar_factor;
        else, config.alpha = config.alpha/config.bar_factor; end
        
        mem = 0;
        count = 0;
        while mem == 0 && count < config.bar_checks
            AD_out = gen_AD(config,des_net,gen_net,z,z_target,1);
            bar_est = flintmax;
            if AD_out.mem == 1, bar_est = max(AD_out.ens); end
            bar_out = max([bar_est, ...
                    get_gen_energy(config,des_net,gen_net,z), ...
                        get_gen_energy(config,des_net,gen_net,z_target)]);
            if bar_out < barrier, barrier = bar_out; a_bar=config.alpha; end
            mem = mem + AD_out.mem;
            count = count+1;
            disp('***');
            disp(count);
            disp(bar_out);
            disp(a_bar);
            disp(config.alpha);
        end
        mem = 1*(mem>0);
    end
    
    if mem == 0, a_border = config.alpha*config.bar_factor;
    else, a_border = config.alpha; end
end