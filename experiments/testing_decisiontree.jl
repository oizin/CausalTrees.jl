using Plots
import Statistics
using CausalTrees

"""
logistic
"""
logistic(α) = exp(α) / (1 + exp(α))


# TEST 1 ######################################################################
# generate data
x = randn(100)
y = rand(100)
y[x .> 0] .+=  2
# plot
scatter(x,y,label="",legend=:bottomright)
# test split finding algorithm: split #1
@time split_i,thres,loss = CausalTrees.evaluate_possible_splits(x[:,1],y,10)
#@time pos_thres = CausalTrees.compute_possible_splits(x[:,1],10)
vline!([thres], label = "split #1")
# test split finding algorithm: split #2
msk = x[:,1] .< 0.0
@time split_i,thres,loss = CausalTrees.evaluate_possible_splits(x[msk,1],y[msk],10)
vline!([thres], label = "split #2")

# TEST 2 #######################################################################
# generate data
x = randn(1000,1)
y = rand(1000)
y[x[:,1] .> 0] .+=  2
y[x[:,1] .> 1] .+=  1
# plot
scatter(x[:,1],y,label="")
# fit tree and predict
@time t1 = CausalTrees.decision_tree(x,y,mse_tau,min_leaf_size=100)
yhat = predict(t1,x)
scatter!(x[:,1],yhat,label="")

# TEST 3 #######################################################################
# generate data
x = randn(1000,5)
e = 6 .* rand(1000)
y = 2 .* x[:,1] .+ e
y[x[:,1] .> 0] .+=  1.5
# plot
scatter(x[:,1],y,label="")
# fit tree and predict
@time t1 = CausalTrees.decision_tree(x,y,mse_tau,min_leaf_size=100)
yhat = predict(t1,x)
scatter!(x[:,1],yhat,label="")

# TEST 4 #######################################################################
# generate data
x = randn(1000,1)
y = rand(1000)
y[x[:,1] .> 0] .+=  2
y[x[:,1] .> 1] .+=  1
# plot
scatter(x[:,1],y,label="",legend=:bottomright)
# fit tree and predict
@time t1 = CausalTrees.decision_tree(x,y,mse_tau,min_leaf_size=50) # not honest
@time t2 = CausalTrees.decision_tree(x,y,mse_tau,min_leaf_size=50,honesty=true) # honest
yhat1 = predict(t1,x)
yhat2 = predict(t2,x)
scatter!(x[:,1],yhat1,label="not honest")
scatter!(x[:,1],yhat2,label="honest")

# TEST 5 #######################################################################
# generate data
x = randn(1000,5)
e = 6 .* rand(1000)
y = 2 .* x[:,1] .+ e
y[x[:,1] .> 0] .+=  1.5
# plot
scatter(x[:,1],y,label="")
# fit random forest and predict
@time rf1 = CausalTrees.random_forest(x,y,mse_tau,n_trees=500,row_subsamp_p=0.8,
    min_leaf_size=30,honesty=true)
yhat = predict(rf1,x)
scatter!(x[:,1],yhat,label="")

# TEST 6 #######################################################################
# binary data
n = 2000
# covariates
x = randn((n,5))
# treatment group
w = rand([0,1],n)
# part of treatment effect
α = 1.8 .* x[:,1] + 0.8 .* x[:,2]
yp = logistic.(α + 0.01 .* randn(n))
y = zeros(n)
rr = rand(n)
y[yp .> rr] .= 1.0
y = reshape(y,n)
# plot
scatter(x[:,1],y,label="")
# fit random forest and predict
@time rf1 = CausalTrees.random_forest(x,y,mse_tau,n_trees=1000,row_subsamp_p=0.8,
    min_leaf_size=30,honesty=false)



yhat = predict(rf1,x)
scatter!(x[:,1],yhat,label="")
scatter!(x[:,1],logistic.(0.6 .* x[:,1] + 0.5 .* x[:,2]),label="")

scatter(yp,yhat,label="")
plot!(Shape([(0,0),(1,1)]),label="")
