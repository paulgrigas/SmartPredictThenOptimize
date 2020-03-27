include("../oracles/assignment_oracle.jl")

using Test

@testset "Assignment Oracle" begin
    C = [2.0 6; 3 5]
    c = reshape(C, 4)
    (z, w) = assignment_oracle(c)
    W = reshape(w, (2,2))
    @test z ≈ 7.0
    @test W ≈ [1 0; 0 1]

    c_ham = assignment_hamming_reduce(c)
    (z2, w2) = assignment_oracle(c_ham)
    W2 = reshape(w2, (2,2))
    @test z2 ≈ 0
    @test W2 ≈ W
end
