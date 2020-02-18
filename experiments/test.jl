
import Statistics

function array_sortperm(x::Array{Float64,2})
    xsp = zeros(Int16,size(x))
    @inbounds for j in 1:size(x)[2]
        if Statistics.std(x[:,j]) > 0.0
            xsp[:,j] = sortperm(x[:,j])
        else
            xsp[:,j] .= -1
        end
    end
    return(xsp)
end

x = randn(10000,100)
@time xsp = array_sortperm(x)
xsp

# how we select data
# the region we use in the lower ssplit is x[xsp[rows,last col],]
# move to this approach!
@time x[xsp[30:300,2],]


# Simulation #1 ---------------------------------------------------------------
x,w,y,tau = gen_trial1(1000)

root = CausalTrees.Node(collect(1:1000),0)
@time CausalTrees.learn_split!(root,x,w,y,mse_tau,10,0.0,4)
root = 0


x1 = x[:,1]
@time sort(x1)

@time x2 = sort(x1)
@time sort!(x1)

x[:,1]

x1

x2

################################################################################

function compute_possible_splits(x)
    x = sort(x)
    n = length(x)
    splits = Vector{Float64}(undef,n-1)
    @inbounds for i in 1:(n-1)
        splits[i] = (x[i] + x[i+1])/2.0
    end
    return(splits)
end

x,w,y,tau = gen_trial1(1000)

@time xs = compute_possible_splits(x[:,3])


function evaluate_possible_splits(x,y)
    oj = sortperm(x)
    yj = y[oj]
    n = length(x)
    # first split
    split = 2
    yl_ = Statistics.mean(yj[1:2])
    yr_ = Statistics.mean(yj[3:end])
    ssl = sum(yj[1:2].^2)
    ssr = sum(yj[3:end].^2)
    vl = ssl/2 - yl_^2
    vr = ssr/(n-2) - yr_^2
    prop_loss = 2*vl + (n-2)*vr
    @inbounds for i in 2:(n-2)
        yl_ = update_average(yl_,yj[i],i)
        yr_ = update_average(yl_,yj[i],n-i)
        ssl = ssl + yj[i]^2
        ssr = ssr - yj[i]^2
        vl = ssl/i - yl_^2
        vr = ssr/(n-i) - yr_^2
        tmp = i*vl + (n-i)*vr
        if tmp > prop_loss
            prop_loss = tmp
            split = i
        end
    end
    return(split,prop_loss)
end

function update_average(current,val,n,remove::Bool=false)
    if remove == false
        update = (n/(n+1))*current + (1/(n+1))*val
    elseif remove == true
        update = (n/(n-1))*current - (1/(n-1))*val
    end
    return(update)
end


println(update_average(Statistics.mean(x[1:10,3]),x[11,3],10))
println(Statistics.mean(x[1:11,3]))

println(update_average(Statistics.mean(x[1:9,3]),x[9,3],9,true))
println(Statistics.mean(x[1:8,3]))

x,w,y,tau = gen_trial1(1000)
@time best = evaluate_possible_splits(x[:,1],y)
println(best)



x = randn(100)
e = rand(100)
y = e
y[x .> 0] .+=  2
scatter(x,y)
