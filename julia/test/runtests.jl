using Test
using MachineLearningCourse

@testset "MachineLearningCourse" begin
    @testset "Lecture01" begin
        @test redirect_stdout(devnull) do
            Lecture01.demo([1,0,0,1]) ≈ [0,1]
        end
        @test redirect_stdout(devnull) do
            Lecture01.demo([0,1,1,0]) ≈ [1,0]
        end
    end

    @testset "Lecture02" begin
        @test_nowarn redirect_stdout(devnull) do
            Lecture02.demo()
        end
    end
end


