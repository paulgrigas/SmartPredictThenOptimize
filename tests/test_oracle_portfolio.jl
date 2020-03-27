include("../oracles/portfolio_oracle.jl")

using Test, Random, LinearAlgebra

@testset "Portfolio Oracle" begin
    Random.seed!(342)

    d = 10
    factors = 4
    F = 0.05*rand(d, factors)

    mu = rand(d)
    r = mu + F*randn(factors) + 0.2*randn(d)

    println("R is : $r")

    Sigma = F*F' + Matrix(0.2I, d, d)

    (z1, w1) = portfolio_simplex_oracle_jump_basic(-r, Sigma, 0.1)

    oracle = portfolio_simplex_jump_setup(Sigma, 0.1)
    (z2, w2) = oracle(-r)
    
    @test isapprox(z1, z2; atol = 0.0001)
    @test isapprox(w1, w2; atol = 0.0001)
end
