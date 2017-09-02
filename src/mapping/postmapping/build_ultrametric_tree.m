function nodes = build_ultrametric_tree(barrier_mat)

    n_mins = size(barrier_mat,1);
    active_nodes = 1:n_mins;
    
    min_energys = diag(barrier_mat)';
    
    nodes = {};
    for i = 1:n_mins
        node.energy = min_energys(i);
        node.ID = i;
        node.children = [];
        node.parent = [];
        nodes{end+1} = node;
    end
    
    %terminate tree construction early if all remaining barriers are infty
    infbar = 0;
    bar_mat = barrier_mat;
    while length(active_nodes)>1 && ~infbar
        min_bar = min(bar_mat(logical(triu(ones(size(bar_mat,1)),1))));
        if min_bar == flintmax
            infbar = 1;
        else
            min_inds = find(bar_mat == min_bar);
            i = mod(min_inds(1)-1,size(bar_mat,1))+1;
            j = floor((min_inds(1)-1)/size(bar_mat,1))+1;
            count_ij = 1;
            while i==j
                count_ij = count_ij+1;
                i = mod(min_inds(count_ij)-1,size(bar_mat,1))+1;
                j = floor((min_inds(count_ij)-1)/size(bar_mat,1))+1;
            end
            bar_mat = consolidate_bar_mat(bar_mat,i,j);
            node.energy = min_bar;
            node.ID = length(nodes) + 1;
            node.children = active_nodes([i,j]);
            node.parent = [];
            nodes{end+1} = node;
            nodes{active_nodes(i)}.parent = length(nodes);
            nodes{active_nodes(j)}.parent = length(nodes);
            active_nodes = [setdiff(active_nodes,active_nodes([i,j])),length(nodes)];           
        end  
    end      
end

function bar_out = consolidate_bar_mat(bar_mat,i,j)
    n = size(bar_mat,1);
    bar_out = zeros(n-1);  
    kept_inds = setdiff(1:n,[i,j]);
    bar_out(1:(n-2),1:(n-2)) = bar_mat(kept_inds,kept_inds);
    bar_out(end,1:(n-2)) = min(bar_mat([i,j],kept_inds));
    bar_out(1:(n-2),end) = bar_out(end,1:(n-2));
    bar_out(n-1,n-1) = min(bar_mat(i,i),bar_mat(j,j));
end