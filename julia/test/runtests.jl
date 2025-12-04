using Test
using Suppressor
using MachineLearningCourse

# Test all lecture modules
@testset "MachineLearningCourse" begin
    
    @testset "Lecture01" begin
        @test @suppress_out(Lecture01.demo([1,0,0,1])) ≈ [0,1]
        @test @suppress_out(Lecture01.demo([0,1,1,0])) ≈ [1,0]
    end
end
