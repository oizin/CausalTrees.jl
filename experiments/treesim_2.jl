using Plots
import Statistics
using CausalTrees


x = randn(1000,2)
e = rand(1000)
y = e
y[x[:,1] .> 0] .+=  2
y[x[:,1] .> 1] .+=  1

scatter(x[:,1],y,label="")
@time t1 = CausalTrees.decision_tree(x,y,mse_tau,10,0.0,3)
yhat = predict(t1,x)
scatter!(x[:,1],yhat,label="")

scatter(x[:,2],y,label="")
scatter!(x[:,2],yhat,label="")

################################################################################
x = randn(100)
e = rand(100)
y = e
y[x .> 0] .+=  2
scatter(x,y)

@time CausalTrees.evaluate_possible_splits(x[:,1],y)

###############################################################################

x = randn(1000,2)
e = rand(1000)
y = 2 .* x[:,1] + randn(1000)

@time t1 = CausalTrees.decision_tree(x,y,mse_tau,5,0.0,4)
yhat = predict(t1,x)

scatter(x[:,1],y,label="")
scatter!(x[:,1],yhat,label="")

scatter(x[:,2],y,label="")
scatter!(x[:,2],yhat,label="")
