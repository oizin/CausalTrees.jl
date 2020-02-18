import Statistics
import StatsBase

#using .Loss

"""
compute_possible_splits

"""
function compute_possible_splits(x)
    x = sort(x)
    n = length(x)
    splits = Vector{Float64}(undef,(n-1))
    @inbounds for i in 1:(n-1)
        splits[i] = (x[i] + x[i+1])/2.0
    end
    return(splits)
end


"""
evaluate_possible_splits

"""
function evaluate_possible_splits(x,y)
    oj = sortperm(x)
    yj = y[oj]
    n = length(x)
    # first split
    split = 2  # actually (x[1] + x[2])/2  x["1.5"]
    yl_ = Statistics.mean(yj[1])
    yr_ = Statistics.mean(yj[2:end])
    best_loss = yl_^2 + (n-1)*yr_^2
    @inbounds for i in 2:(n-1)
        yl_ = update_average(yl_,yj[i],i)
        yr_ = update_average(yr_,yj[i],n-i,true)
        loss_i = i*yl_^2 + (n-i)*yr_^2
        if loss_i > best_loss
            best_loss = loss_i
            split = i
        end
    end
    return(split,best_loss)
end


"""
Update an average based on a newly observed value

"""
function update_average(current,val,n,remove::Bool=false)
    if remove == false
        update = (n/(n+1))*current + (1/(n+1))*val
    elseif remove == true
        update = (n/(n-1))*current - (1/(n-1))*val
    end
    return(update)
end


"""
learn a split given two datasets X and y

Arguments

node: the node to split
X: feature matrix
y: response
loss: loss function
min_leaf_size: minimum number of unique values in a leaf

Note: aim is to maximise the loss function
"""
function learn_split!(
    node                :: Node,
    X                   :: Array{Float64,2},
    y                   :: Array{Float64,1},
    loss                :: Function,
    min_leaf_size       :: Int64,
    min_loss_increase   :: Float64,
    max_depth           :: Int64
    )

    region = node.region
    n_samples = length(region)
    n_features = size(X)[2]

    # split parameters
    feature = 0
    threshold = NaN

    # X and y values in partition region
    Xp = X[region,:]
    yp = y[region]

    # prediction and loss in partition region
    mu_ = Statistics.mean(yp)
    current_loss = n_samples*mu_^2

    #println(n_samples," ",min_leaf_size, node.depth < max_depth)

    # test whether maximum depth reached
    if ((node.depth < max_depth) & (n_samples > min_leaf_size))
        # for each column evaluate all possible split points
        # outer loop: loop over columns
        @inbounds for j in 1:n_features
            xpj = Xp[:,j]
            spj = compute_possible_splits(xpj)
            split,prop_loss = evaluate_possible_splits(xpj,yp)
            if prop_loss > current_loss
                feature = j
                current_loss = prop_loss
                threshold = spj[split-1]
                global region_l = region[xpj .< threshold]
                global region_r = region[xpj .> threshold]
            end
        end
    end

    if feature != 0
        # modify the node
        node.value = mu_  # placeholder for pruning
        node.is_leaf = false
        node.feature = feature
        node.threshold = threshold
        node.l = Node(region_l,node.depth+1)
        node.r = Node(region_r,node.depth+1)
    else
        # modify the node
        node.value = mu_
        node.is_leaf = true
    end
end

"""
learn_tree :
learn a decision tree

X: feature matrix
y: response
loss: loss function
min_leaf_size: minimum number of unique values in a leaf
max_depth: maximum tree depth

Note: aim is to maximise the loss function
"""
function learn_tree(
    X                   :: Array{Float64,2},
    y                   :: Array{Float64,1},
    loss                :: Function,
    min_leaf_size       :: Int64,
    min_loss_increase   :: Float64,
    max_depth           :: Int64
    )

    n_samples, n_features = size(X)
    indX = collect(1:n_samples)

    # root node
    root = Node(indX,0)
    stack = Node[root] # here tree.Node indicates array type
    ids = Array{Int64,1}()

    while length(stack) > 0
        node = pop!(stack)
        node.id = node_id(ids)
        append!(ids,node.id)
        learn_split!(node,X,y,loss,min_leaf_size,min_loss_increase,max_depth)  # in place modification of node.l, node.r
        if !node.is_leaf
            push!(stack,node.r)
            push!(stack,node.l)
        end
    end
    return(root)
end

"""
estimate_leaf_value :

"""
function learn_leaf_value(
    tree                :: Node,
    X                   :: Array{Float64,2},
    y                   :: Array{Float64,1}
    )

    n_samples, n_features = size(X)
    id_vec = match_to_id(tree,X)

    id_to_val = Dict()  # add types

    for id in unique(id_vec)
        yp = y[id .== id_vec]
        Xp = X[id .== id_vec,:]
        X_id = Statistics.mean(Xp,dims=1)
        id_to_val[id] = [Statistics.mean(yp),X_id]
    end

    for id in unique(id_vec)
        node = tree
        while node.is_leaf == false
            if id_to_val[id][2][node.feature] < node.threshold
                node = node.l
            else
                node = node.r
            end
        end
        node.value = id_to_val[id][1]
    end
    return(tree)
end

"""
fit a decision tree

Arguments

structure_p : the proportion of the data used in estimating the tree structure (i.e. nodes)
"""
function decision_tree(
    X                   :: Array{Float64,2},
    y                   :: Array{Float64,1},
    loss                :: Function,
    min_leaf_size       :: Int64,
    min_loss_increase   :: Float64,
    max_depth           :: Int64,
    honesty             :: Bool=false,
    structure_p         :: Float64=0.5
    )

    # dataset dimensions
    n_samples, n_features = size(X)

    if (honesty == true)
        indX = collect(1:n_samples)

        # indexes for the structural and leaf parts of the tree
        n_struct = convert(Int64,floor(structure_p*n_samples))
        indS = StatsBase.sample(indX,n_struct,replace=false)
        indH = setdiff(indX,indS)

        # structure stage
        XS = X[indS,:]
        yS = y[indS]
        tree = learn_tree(XS,yS,loss,min_leaf_size,
                          min_loss_increase,max_depth)

        # estimate leaf values
        XH = X[indH,:]
        yH = y[indH]
        tree = learn_leaf_value(tree,XH,yH)
    elseif (honesty == false)
        tree = learn_tree(X,y,loss,min_leaf_size,
                          min_loss_increase,max_depth)
    end

    # remove region from tree
    recursive_action(tree,remove_region!)
    # return as type Tree
    return(Tree(tree,n_features))

end

"""
causal_forest : learn a causal forest

"""
function random_forest(
    X                   :: Array{Float64,2},
    y                   :: Array{Float64,1},
    loss                :: Function,
    n_trees             :: Int64,
    col_subsamp_p       :: Float64,
    row_subsamp_p       :: Float64,
    row_resample        :: Bool,
    min_leaf_size       :: Int64,
    min_loss_increase   :: Float64,
    max_depth           :: Int64,
    honesty             :: Bool=false,
    structure_p         :: Float64=0.5
    )

    # dataset dimensions
    n_samples, n_features = size(X)
    ind_r = collect(1:n_samples)
    ind_c = collect(1:n_features)

    # column and row subsampling number
    col_subsamp_n = convert(Int64,floor(col_subsamp_p*n_features))
    row_subsamp_n = convert(Int64,floor(row_subsamp_p*n_samples))

    # initialise empty array for trees and column indices
    trees = Array{CausalTrees.Tree,1}(undef,n_trees)
    feature_index = Array{Int64,2}(undef,(n_trees,col_subsamp_n))

    for t in 1:n_trees
        # subsamples
        ind_r_t = StatsBase.sample(1:n_samples,row_subsamp_n,replace=row_resample)
        ind_c_t = StatsBase.sample(1:n_features,col_subsamp_n,replace=false)
        Xt = X[ind_r_t,ind_c_t]
        yt = y[ind_r_t]
        # train forest
        trees[t] = causal_tree(Xt,yt,loss,min_leaf_size,min_loss_increase,
            max_depth,honesty,structure_p)
        feature_index[t,:] = ind_c_t
    end
    return(Forest(trees,feature_index,n_trees,n_features))
end
