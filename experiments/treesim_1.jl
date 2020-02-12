using Plots
import Statistics
using CausalTrees

# Functions to generate data --------------------------------------------------
"""
Generate RCT like data, with heterogeneity in the treatment effect
n: number of samples (total)
pnoise: number of noise variable
"""
function gen_trial1(n::Int,pnoise::Int=0)
    # covariates
    x = randn((n,3+pnoise))
    # treatment group
    w = rand([0,1],n)
    # The treatment effect: E(Y1-Y0|X)
    tau =  2 .+ 3.35 .* x[:,1] + 0.8 .* x[:,1].^2 + 0.3 .* x[:,1].^3 + 1.4 .* x[:,2] .* x[:,3]
    # The potential outcomes
    y0 = 1 .* x[:,1] + x[:,2] + randn(n)
    y1 = 1 .* x[:,1] + x[:,2] + tau + randn(n)
    truth = y1 .- y0
    # observed outcome
    y = (1 .- w) .* y0 + w .* y1
    y = reshape(y,n)
    (x,w,y,tau)
end

# Simulation #1 ---------------------------------------------------------------
x,w,y,tau = gen_trial1(1000)


# causal_tree(X,w,y,loss,min_leaf_size,min_loss_increase,max_depth,honesty,structure_p)
@time t1 = causal_tree(x,w,y,mse_tau,10,0.0,3)
tauhat1 = CausalTrees.predict(t1,x)
Plots.scatter(x[:,1], tau,label="truth",legend=:topleft)
Plots.scatter!(x[:,1],tauhat1,label="not honest")


@time t2 = causal_tree(x,w,y,mse_tau,10,0.0,8,true,0.5)




tauhat2 = CausalTree.predict(t2,x)
nh_error = sqrt(Statistics.mean((tauhat1 .- truth).^2))
h_error = sqrt(Statistics.mean((tauhat2 .- truth).^2))
println(nh_error)
println(h_error)
println(h_error/nh_error)
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
