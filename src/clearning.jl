import Statistics
import StatsBase


"""
Calculate group mean

"""
function group_mean(
    y                   :: Array{Float64,1},
    w                   :: Array{Int64,1},
    group               :: Int64
    )

    yg_ = Statistics.mean(y[w .== group])
    return(yg_)
end

"""
Evaluate potential splits on treatment effect\n
Arguments:\n
- x:
- w:
- y:
- min_leaf_size:
"""
function evaluate_possible_splits(
    x                   :: Array{Float64,1},
    w                   :: Array{Int64,1},
    y                   :: Array{Float64,1},
    min_leaf_size       :: Int
    )

    oj = sortperm(x)
    yj = y[oj]
    wj = w[oj]
    n = length(w)
    # first split
    split = min_leaf_size

    # left
    l_indX = 1:min_leaf_size
    y0l_ = group_mean(yj[l_indX],wj[l_indX],0)
    y1l_ = group_mean(yj[l_indX],wj[l_indX],1)
    taul_ = y1l_ - y0l_
    # right
    r_indX = (min_leaf_size+1):n
    y0r_ = group_mean(yj[r_indX],wj[r_indX],0)
    y1r_ = group_mean(yj[r_indX],wj[r_indX],1)
    taur_ = y1r_ - y0r_

    n0l = sum(w[l_indX] .== 0)
    n1l = sum(w[l_indX] .== 1)
    n0r = sum(w[r_indX] .== 0)
    n1r = sum(w[r_indX] .== 1)

    start = min_leaf_size
    while n0l == 0 | n1l == 0 | n0r == 0 | n1r == 0
        start += 1
    end

    best_loss = (1/n)*(start*taul_^2 + (n-start)*taur_^2) # update to function calling

    @inbounds for i in (min_leaf_size+1):(n-min_leaf_size+1)
        if wj[i] == 0
            n0l += 1
            n0r -= 1
            y0l_ = update_average(y0l_,yj[i],n0l)
            y0r_ = update_average(y0r_,yj[i],n0r,true)
        elseif wj[i] == 1
            n1l += 1
            n1r -= 1
            y1l_ = update_average(y1l_,yj[i],n1l)
            y1r_ = update_average(y1r_,yj[i],n1r,true)
        end
        # update treatment effect and loss
        taul_ = y1l_ - y0l_
        taur_ = y1r_ - y0r_
        loss_i = (1/n)*(i*taul_^2 + (n-i)*taur_^2)
        if loss_i > best_loss
            best_loss = loss_i
            split = i
        end
    end
    # compute threshold
    threshold = (x[oj][split] + x[oj][split+1])/2.0
    # return
    return(split,threshold,best_loss)
end


"""
learn_split :
learn a split given X, w and y

node: the node to split
X: feature matrix
w: vector of treatment groups
y: response
loss: loss function
min_leaf_size: minimum number of unique values in a leaf

Note: aim is to maximise the loss function
"""
function learn_split!(
    node                :: Node,
    X                   :: Array{Float64,2},
    w                   :: Array{Int64,1},
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

    # X,w and y values in partition region
    Xp = X[region,:]
    wp = w[region]
    yp = y[region]

    # treatment effect and loss in partition region
    tau_ = ate(yp,wp)
    current_loss = (1/n_samples) * tau_^2

    # test whether maximum depth reached
    if ((node.depth < max_depth) & (n_samples > min_leaf_size))
        # for each column evaluate all possible split points
        # outer loop: loop over columns
        @inbounds for j in 1:n_features
            xpj = Xp[:,j]
            #spj = compute_possible_splits(xpj,min_leaf_size)
            split,prop_threshold,prop_loss = evaluate_possible_splits(xpj,wp,yp,min_leaf_size)
            if prop_loss > current_loss
                feature = j
                current_loss = prop_loss
                threshold = prop_threshold
                global region_l = region[xpj .< threshold]
                global region_r = region[xpj .> threshold]
            end
        end
    end

    if feature != 0
        # modify the node
        node.value = tau_  # placeholder for pruning
        node.is_leaf = false
        node.feature = feature
        node.threshold = threshold
        node.l = Node(region_l,node.depth+1)
        node.r = Node(region_r,node.depth+1)
    else
        # modify the node
        node.value = tau_
        node.is_leaf = true
    end
end

"""
learn a causal tree

X: feature matrix
y: response
w: treatment group
loss: loss function
min_leaf_size: minimum number of unique values in a leaf
max_depth: maximum tree depth

Note: aim is to maximise the loss function
"""
function learn_tree(
    X                   :: Array{Float64,2},
    w                   :: Array{Int64,1},
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
        learn_split!(node,X,w,y,loss,min_leaf_size,min_loss_increase,max_depth)  # in place modification of node.l, node.r
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
    w                   :: Array{Int64,1},
    y                   :: Array{Float64,1}
    )

    n_samples, n_features = size(X)
    id_vec = match_to_id(tree,X)

    id_to_val = Dict()  # add types

    for id in unique(id_vec)
        yp = y[id .== id_vec]
        wp = w[id .== id_vec]
        Xp = X[id .== id_vec,:]
        X_id = Statistics.mean(Xp,dims=1)
        id_to_val[id] = [ate(yp,wp),X_id]
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
Fit a causal tree\n
Arguments:\n
- structure_p : the proportion of the data used in estimating the tree structure (i.e. nodes)
"""
function causal_tree(
    X                   :: Array{Float64,2},
    w                   :: Array{Int64,1},
    y                   :: Array{Float64,1},
    loss                :: Function;
    min_leaf_size       :: Int64=20,
    min_loss_increase   :: Float64=0.0,
    max_depth           :: Int64=3,
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
        wS = w[indS]
        yS = y[indS]
        tree = learn_tree(XS,wS,yS,loss,min_leaf_size,
                          min_loss_increase,max_depth)

        # estimate leaf values
        XH = X[indH,:]
        yH = y[indH]
        wH = w[indH]
        tree = learn_leaf_value(tree,XH,wH,yH)
    elseif (honesty == false)
        tree = learn_tree(X,w,y,loss,min_leaf_size,
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
function causal_forest(
    X                   :: Array{Float64,2},
    w                   :: Array{Int64,1},
    y                   :: Array{Float64,1},
    loss                :: Function;
    n_trees             :: Int64=100,
    col_subsamp_p       :: Float64=1.0,
    row_subsamp_p       :: Float64=1.0,
    row_resample        :: Bool=true,
    min_leaf_size       :: Int64=10,
    min_loss_increase   :: Float64=0.0,
    max_depth           :: Int64=3,
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
        wt = w[ind_r_t]
        yt = y[ind_r_t]
        # train forest
        trees[t] = causal_tree(Xt,wt,yt,loss,
            min_leaf_size=min_leaf_size,
            min_loss_increase=min_loss_increase,
            max_depth=max_depth,
            honesty=honesty,
            structure_p=structure_p)
        feature_index[t,:] = ind_c_t
    end
    return(Forest(trees,feature_index,n_trees,n_features))
end
