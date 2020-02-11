import Plots
import Statistics
include("main.jl")
import .CausalTree

# A single tree
n = 1000
x = randn((n,3)); w = rand([0,1],n)
tau =  2 .+ 3.35 .* x[:,1] + 0.8 .* x[:,1].^2 + 0.3 .* x[:,1].^3 + 1.4 .* x[:,2] .* x[:,3]
y0 = 1 .* x[:,1] + x[:,2] + randn(n)
y1 = 1 .* x[:,1] + x[:,2] + tau + randn(n)
y = (1 .- w) .* y0 + w .* y1
y = reshape(y,n)
@time t1 = CausalTree.causal_tree(x,w,y,CausalTree.mse_tau,10,0.0,8)
@time t2 = CausalTree.causal_tree(x,w,y,CausalTree.mse_tau,10,0.0,8,true,0.7)
tauhat1 = CausalTree.predict(t1,x); tauhat2 = CausalTree.predict(t2,x)
truth = y1 .- y0
nh_error = sqrt(Statistics.mean((tauhat1 .- truth).^2))
h_error = sqrt(Statistics.mean((tauhat2 .- truth).^2))
println(nh_error)
println(h_error)
println(h_error/nh_error)
Plots.scatter(x[:,1], truth,label="truth",legend=:topleft)
Plots.scatter!(x[:,1],tauhat1,label="not honest")
Plots.scatter!(x[:,1],tauhat2,label="honest")

# printing
tmp = CausalTree.tree_to_array(t1.tree)
CausalTree.print(t1.tree)

# plot the tree
CausalTree.plot(t1)

# A forest
n = 1000
x = randn((n,10)); w = rand([0,1],n)
tau =  2 .+ 3.35 .* x[:,1] + 0.8 .* x[:,1].^2 + 0.3 .* x[:,1].^3
y0 = 1 .* x[:,1] + randn(n)
y1 = 1 .* x[:,1] + tau + randn(n)
truth = y1 .- y0
y = (1 .- w) .* y0 + w .* y1
y = reshape(y,n)
@time t1 = CausalTree.causal_tree(x,w,y,CausalTree.mse_tau,10,0.0,8)
tauhat1 = CausalTree.predict(t1,x)
@time f1 = CausalTree.causal_forest(x,w,y,CausalTree.mse_tau,20,0.8,0.8,false,10,0.0,8)
@time tauhat3 = CausalTree.predict(f1,x)

# current forest algorithm is wrong!!!!

Plots.scatter(x[:,1], truth,label="truth",legend=:topleft)
Plots.scatter!(x[:,1],tauhat1,label="tree")
Plots.scatter!(x[:,1],tauhat3,label="forest")


CausalTree.predict(f1.trees[1],x)
CausalTree.predict(f1.trees[2],x)
CausalTree.predict(f1.trees[3],x)
CausalTree.predict(f1.trees[4],x)
CausalTree.predict(f1.trees[5],x)


function causal_forest(
    X::Array{Float64,2},
    w::Array{Int64,1},
    y::Array{Float64,1},
    loss::Function,
    n_trees::Int64,
    col_subsamp_p::Float64,
    row_subsamp_p::Float64,
    row_resample::Bool,
    min_leaf_size::Int64,
    min_loss_increase::Float64,
    max_depth::Int64,
    honesty::Bool=false,
    structure_p::Float64=0.5)
