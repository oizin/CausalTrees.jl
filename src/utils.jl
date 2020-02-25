using PrettyTables

"""
match_to_id :
Match each row of a sample to a leaf id (key)
"""
function match_to_id(tree,X::Array{Float64,2})
    n_samples, n_features = size(X)
    id_vec = Array{Int64,1}()

    for i in 1:n_samples
        node = tree
        while node.is_leaf == false
            if X[i,node.feature] < node.threshold
                node = node.l
            else
                node = node.r
            end
        end
        push!(id_vec,node.id)
    end
    return(id_vec)
end

"""
recursive_action :
Perform a recursive action (e.g. print) over the nodes of a tree
"""
function recursive_action(tree,action::Function)
    if tree.is_leaf == false
        action(tree)
        recursive_action(tree.l,action)
        recursive_action(tree.r,action)
    else
        action(tree)
    end
end

"""
remove_region :
Remove region index information from a node
"""
function remove_region!(node)
    node.n = size(node.region)[1]
    node.region = nothing
end

"""
tree_to_array :
convert a tree to an array
"""
function tree_to_array(tree,parent=-1,tree_tab=Array{Union{Int64,Float64,Bool},2}(undef,0,8))
    if tree.is_leaf == false
        tree_tab = vcat(Union{Int64,Float64,Bool}[tree.id parent tree.is_leaf tree.value tree.feature tree.threshold tree.depth tree.n],
                    tree_to_array(tree.l,tree.id,tree_tab),
                    tree_to_array(tree.r,tree.id,tree_tab))
    else
        tree_tab = Union{Int64,Float64,Bool}[tree.id parent tree.is_leaf tree.value NaN NaN tree.depth tree.n]
    end
    return(tree_tab)
end

"""
print a tree

"""
function print(
    tree
    ;digits=2)
    tree = tree.tree
    header = ["id", "parent","leaf","value","feature","threshold","depth","n"]
    tree_tab = tree_to_array(tree)
    tree_tab = tree_tab[sortperm(tree_tab[:,7]),:]
    formatter = Dict(4 => (v,i)-> round(v;digits=digits),
                    6 => (v,i)-> round(v;digits=digits))
    pretty_table(tree_tab,header;formatter=formatter)
end

"""
node_id :
Generate a unique ID for each node
"""
function node_id(ids::Array{Int64,1})
    if (length(ids) == 0)
        return(0)
    else
        return(1 + maximum(ids))
    end
end
