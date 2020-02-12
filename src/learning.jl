import Statistics
import StatsBase

#using .Loss

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
    println("called...")
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
    current_loss = loss(tau_)

    # test whether maximum depth reached
    if (node.depth <= max_depth)
        # for each column evaluate all possible split points
        # outer loop: loop over columns
        for j in 1:n_features
            xpj = Xp[:,j]
            xsj = sort(xpj)
            # inner loop: loop over unique row values of a column
            for i in 2:(n_samples)
                # don't test the same threshold more than once
                if i > 1
                    if xsj[i] == xsj[i-1]
                        continue
                    end
                end
                proposal = xsj[i]  # proposed threshold
                # construct lhs
                mskl = xpj .< proposal
                # construct rhs
                mskr = xpj .>= proposal
                # test whether to continue
                if ((sum(wp[mskl] .== 0) < min_leaf_size) ||
                    (sum(wp[mskl]  .== 1) < min_leaf_size) ||
                    (sum(wp[mskr]  .== 0) < min_leaf_size) ||
                    (sum(wp[mskr]  .== 1) < min_leaf_size))
                    continue  # fail => move to next split point
                end
                # evaluate splits
                taur_ = ate(yp[mskr],wp[mskr])
                taul_ = ate(yp[mskl],wp[mskl])
                prop_loss = loss(taul_) + loss(taur_)
                if prop_loss > (current_loss + min_loss_increase)
                    feature = j
                    threshold = proposal
                    current_loss = prop_loss
                    global region_l = region[mskl]
                    global region_r = region[mskr]
                end
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
learn_tree :
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
causal_tree :
structure_p : the proportion of the data used in estimating the tree structure (i.e. nodes)
"""
function causal_tree(
    X                   :: Array{Float64,2},
    w                   :: Array{Int64,1},
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
        wt = w[ind_r_t]
        yt = y[ind_r_t]
        # train forest
        trees[t] = causal_tree(Xt,wt,yt,loss,min_leaf_size,min_loss_increase,
            max_depth,honesty,structure_p)
        feature_index[t,:] = ind_c_t
    end
    return(Forest(trees,feature_index,n_trees,n_features))
end
