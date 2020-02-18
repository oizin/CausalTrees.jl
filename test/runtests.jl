using CausalTrees
using Test

@testset "CausalTrees.jl" begin
    @test CausalTrees.update_average(2.5,4.0,2) == 3.0
    @test CausalTrees.update_average(3.0,4.0,3,true) == 2.5
end
