import Statistics

"""
node :
Node object for constructing tree
"""
mutable struct Node
    id              :: Int64
    feature         :: Int64
    threshold       :: Number
    l               :: Node
    r               :: Node
    region          :: Union{Array{Int64,1},Nothing}
    depth           :: Int64
    is_leaf         :: Bool
    value           :: Any

    function Node(region,depth)
        node = new()
        node.depth = depth
        node.region = region
        return(node)
    end
end

"""
CausalTree :
Causal tree object
"""
struct Tree
    tree            :: Node
    n_features      :: Int64
end

"""
predict :
"""
function predict(
    tree            :: Tree,
    X               :: Array{Float64,2}
    )

    tau_ =  Array{Float64,1}()
    n_samples, n_features = size(X)
    tree = tree.tree
    for i in 1:n_samples
        node = tree
        while node.is_leaf == false
            if X[i,node.feature] < node.threshold
                node = node.l
            else
                node = node.r
            end
        end
        push!(tau_,node.value)
    end
    return(tau_)
end


"""
CausalForest :
Causal forest object
"""
struct Forest
    trees           :: Array{Tree,1}
    feature_index   :: Any
    n_trees         :: Int64
    n_features      :: Int64
end

"""
predict :
"""
function predict(
    forest          :: Forest,
    X               :: Array{Float64,2}
    )

    # initialise prediction arrays
    n_samples, n_features = size(X)
    tau_m_ = Array{Float64,2}(undef,(n_samples,forest.n_trees))
    tau_ = Array{Float64,1}(undef,n_samples)

    # predictions from individual trees
    for t in 1:forest.n_trees
        tau_m_[:,t] = predict(forest.trees[t],X[:,forest.feature_index[t,:]])
    end

    # average across trees (columns) for individual predictions
    for i in 1:n_samples
        tau_[i] = Statistics.mean(tau_m_[i,:])
    end

    return(tau_)
end


"""
summary :
Summary of a tree
"""
# function summary(tree)
# end
