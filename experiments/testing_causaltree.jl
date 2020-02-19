using Plots
import Statistics
using CausalTrees

# Functions to generate data ##################################################
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

# TEST 1 ######################################################################
# generate data
x,w,y,tau = gen_trial1(200)
# plot
scatter(x[:,1],y,colour=w .+ 1,label="",legend=:bottomright)
# test split finding algorithm: split #1
@time split_i,thres,loss = CausalTrees.evaluate_possible_splits(x[:,1],w,y,10)
#@time pos_thres = CausalTrees.compute_possible_splits(x[:,1],10)
vline!([thres], label = "split #1")
# test split finding algorithm: split #2
msk = x[:,1] .< thres
@time split_i,thres,loss = CausalTrees.evaluate_possible_splits(x[msk,1],w[msk],y[msk],10)
vline!([thres], label = "split #2")

# TEST 2 #######################################################################
# generate data
x,w,y,tau = gen_trial1(200)
# plot
scatter(x[:,1],tau,color= w,label="")
# fit tree and predict
@time t1 = causal_tree(x,w,y,mse_tau,min_leaf_size=10)
yhat = predict(t1,x)
scatter!(x[:,1],yhat,label="",markershape=:hline,markercolor=:red)

# TEST 3 #######################################################################
# generate data
x,w,y,tau = gen_trial1(200)
# plot
scatter(x[:,1],tau,color= w,label="")
# fit tree and predict
@time t1 = causal_tree(x,w,y,mse_tau,min_leaf_size=10)
@time t2 = causal_tree(x,w,y,mse_tau,min_leaf_size=10,honesty=true)
yhat1 = predict(t1,x)
yhat2 = predict(t2,x)
scatter!(x[:,1],yhat1,label="",markershape=:rect,markercolor=:red,markersize=2)
scatter!(x[:,1],yhat2,label="",markershape=:rect,markercolor=:blue,markersize=2)

# TEST 4 #######################################################################
# generate data
x,w,y,tau = gen_trial1(300)
# plot
scatter(x[:,1],tau,label="")
# fit random forest and predict
@time rf1 = causal_forest(x,w,y,mse_tau,n_trees=500,row_subsamp_p=0.8,
    min_leaf_size=30,honesty=true)
yhat = predict(rf1,x)
scatter!(x[:,1],yhat,label="")
