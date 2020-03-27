include("../oracles/shortest_path_oracle.jl")

using Test

@testset "Shortest Path Oracle" begin
    sources = [1; 1; 2; 3]
    destinations = [2; 3; 4; 4]
    start_node = 1
    end_node = 4
    oracle = sp_flow_jump_setup(sources, destinations, start_node, end_node)

    weights = [4.0; 1; 4; 6]
    (z, w) = oracle(weights)
    @test z ≈ 7.0
    @test w ≈ [0, 1.0, 0, 1.0]

    weights2 = [2.5; 2.0; 1.1; 3.2]
    (z2, w2) = oracle(weights2)
    @test z2 ≈ 3.6
    @test w2 ≈ [1.0, 0, 1.0, 0]


    # Try a 5 edge case
    sources = [1; 1; 2; 3; 3]
    destinations = [2; 3; 4; 4; 2]
    start_node = 1
    end_node = 4
    oracle = sp_flow_jump_setup(sources, destinations, start_node, end_node)

    weights = [4.0; 1; 4; 6; 3]
    (z, w) = oracle(weights)
    @test z ≈ 7.0
    @test w ≈ [0, 1.0, 0, 1.0, 0]

    weights2 = [4.0; 1; 4; 6; 1]
    (z2, w2) = oracle(weights2)
    @test z2 ≈ 6.0
    @test w2 ≈ [0, 1.0, 1.0, 0, 1.0]


    # Big grid test
    sources, destinations = convert_grid_to_list(20,20)
    start_node, end_node = 1, 400
    oracle = sp_flow_jump_setup(sources, destinations, start_node, end_node)
    weights = ones(1520)
    (z, w) = oracle(weights)
    @test z ≈ 38.0
end
