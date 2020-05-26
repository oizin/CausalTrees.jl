using CausalTrees
using Plots
import Statistics

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
    tau =  2 .+ 4.35 .* x[:,1] + 0.8 .* x[:,1].^2 + 0.3 .* x[:,1].^3 + 1.4 .* x[:,2] .* x[:,3]
    # The potential outcomes
    y0 = 1 .* x[:,1] + x[:,2] + randn(n)
    y1 = 1 .* x[:,1] + x[:,2] + tau + randn(n)
    truth = y1 .- y0
    # observed outcome
    y = (1 .- w) .* y0 + w .* y1
    y = reshape(y,n)
    (x,w,y,tau)
end


"""
logistic
"""
logistic(α) = exp(α) / (1 + exp(α))


"""
Generate RCT like data, with heterogeneity in the treatment effect
n: number of samples (total)
pnoise: number of noise variable
"""
function gen_trial2(n::Int,pnoise::Int=0)
    # covariates
    x = randn((n,3+pnoise))
    # treatment group
    w = rand([0,1],n)
    # part of treatment effect
    α = 2 .* x[:,1] + 0.5 .* x[:,2] + 0.7 .* w
    τ = 0.4 .* x[:,1]
    y0 = logistic.(α)
    y1 = logistic.(α) .+ logistic.(τ + 0.1 .* randn(n))
    truth = y1 .- y0
    # observed outcome
    ypr = (1 .- w) .* y0 + w .* y1
    y = zeros(n)
    rr = rand(n)
    y[ypr .> rr] .= 1.0
    y = reshape(y,n)
    (x,w,y,ypr)
end

# TEST 1 ######################################################################
# generate data
x,w,y,tau = gen_trial1(200)
# plot
plotly()
scatter(x[:,1],y,colour=w .+ 1,label="σ",legend=:bottomright)
# test split finding algorithm: split #1
@time split_i,thres,loss = CausalTrees.evaluate_possible_splits(x[:,1],w,y,10)
#@time pos_thres = CausalTrees.compute_possible_splits(x[:,1],10)
vline!([thres], label = L"split #1")
# test split finding algorithm: split #2
msk = x[:,1] .< thres
@time split_i,thres,loss = CausalTrees.evaluate_possible_splits(x[msk,1],w[msk],y[msk],10)
vline!([thres], label = "split #2")

# TEST 2 #######################################################################
# generate data
x,w,y,tau = gen_trial1(200)
# plot
scatter(x[:,1],tau,color= w,label= "")
# fit tree and predict
@time t1 = causal_tree(x,w,y,mse_tau,min_leaf_size=10)
yhat = predict(t1,x)
Plots.scatter!(x[:,1],yhat,label="",markershape=:hline,markercolor=:red)
# plot the tree
@time CausalTrees.plot(t1,leaf_size=5,split_xsize=2,split_ysize=1)
# print the results
CausalTrees.print(t1)

# TEST 3 #######################################################################
# generate data
x,w,y,tau = gen_trial1(200)
# plot
scatter(x[:,1],tau,color= w,label="")
# fit tree and predict
@time t1 = causal_tree(x,w,y,mse_tau,min_leaf_size=20)
@time t2 = causal_tree(x,w,y,mse_tau,min_leaf_size=20,honesty=true)
yhat1 = predict(t1,x)
yhat2 = predict(t2,x)
scatter!(x[:,1],yhat1,label="not honest")
scatter!(x[:,1],yhat2,label="honest")

# TEST 4 #######################################################################
# generate data
x,w,y,tau = gen_trial1(400)
# plot
scatter(x[:,1],tau,colour=w,label="",legend=:bottomright)
hline!([0],label="null effect")
# fit random forest and predict
@time rf1 = causal_forest(x,w,y,mse_tau,n_trees=500,row_subsamp_p=0.8,
    min_leaf_size=20)
yhat = predict(rf1,x)
scatter!(x[:,1],yhat,label="causal forest")

# TEST 4 #######################################################################
import Random
Random.seed!(1234)
# generate data (as above with larger dataset)
x,w,y,tau = gen_trial1(1000)
# plot
scatter(x[:,1],tau,colour=w,label="",legend=:bottomright,markersize=2)
hline!([0],label="null effect")
# fit random forest and predict
@time t1 = causal_tree(x,w,y,mse_tau,min_leaf_size=10,max_depth=5)
@time rf1 = causal_forest(x,w,y,mse_tau,n_trees=100,row_subsamp_p=0.5,min_leaf_size=20,max_depth=10)
tauhat_rf = predict(rf1,x)
tauhat_t = predict(t1,x)
scatter!(x[:,1],tauhat_rf,label="causal forest",colour=:lightblue)
scatter!(x[:,1],tauhat_t,label="causal tree",colour=:pink)
ate_ = CausalTrees.ate(y,w)
hline!([ate_],label="ate")

# TEST 5 #######################################################################
# binary data
n = 5000
# covariates
x = randn((n,5))
# treatment group
w = rand([0,1],n)
# part of treatment effect
α = 0.6 .* x[:,1] + 0.7 .* x[:,2]
τ = 0.4 .+ 0.4 .* x[:,1]
y0 = logistic.(α + 0.01 .* randn(n))
y1 = logistic.(α + τ + 0.01 .* randn(n))
scatter(x[:,1],y1 .- y0)
ypr = (1 .- w) .* y0 + w .* y1
y = zeros(n)
rr = rand(n)
y[ypr .> rr] .= 1.0
y = reshape(y,n)

println("True: ",Statistics.mean(y1 .- y0))

println("Est: ",Statistics.mean(y[w .== 1]) - Statistics.mean(y[w .== 0]))
# you need to add w to x for this to work!!!!!!

# plots
#histogram(tau)
scatter(x[:,1],y1 .- y0,colour=w,label="",legend=:bottomright,markersize=5)
# fit random forest and predict
@time t1 = causal_tree(x,w,y,mse_tau,min_leaf_size=40,max_depth=3,honesty=true)
tauhat_t = predict(t1,x)
scatter!(x[:,1],tauhat_t,label="causal tree",colour=:pink)
ate_ = CausalTrees.ate(y,w)
println(ate_," ", Statistics.mean(tau))
hline!([ate_],label="ate")
