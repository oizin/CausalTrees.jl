
import Statistics

# All loss functions should be of the form
# 

"""
mse_tau
Loss function for optimising a model to mean square error
y: response
y_: prediction

Note: this is the negative mean square error
"""
function mse_tau(tau_::Float64,n::Int)
    return(n*(tau_^2))
end

"""
emse_loss
Loss function for optimising a model to expected mean square error
y: response
y_: prediction
"""
function emse_tau(y,y_::Float64)
    n_sample = length(y)
    return(n_sample*(y_^2 - 2*Statistics.var(y)))
end

"""
usage: treatment_effect([3,5,4,3,6,7,5],[1,1,1,1,0,0,0])
"""
function ate(y::Array{Float64,1},w::Array{Int64,1})
    y1 = Statistics.mean(y[w .== 1])
    y0 = Statistics.mean(y[w .== 0])
    y1 - y0
end
