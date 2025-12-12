using Test
using MachineLearningCourse

@testset "MachineLearningCourse" begin
    @testset "Lecture01" begin
        println("Testing Lecture01...")
        @test redirect_stdout(devnull) do
            Lecture01.demo([1,0,0,1]) ≈ [0,1]
        end
        @test redirect_stdout(devnull) do
            Lecture01.demo([0,1,1,0]) ≈ [1,0]
        end
    end

    @testset "Lecture02" begin
        println("Testing Lecture02...")
        @test_nowarn redirect_stdout(devnull) do
            Lecture02.demo()
        end
    end

    @testset "Lecture03" begin
        println("Testing Lecture03...")
        @test_nowarn redirect_stdout(devnull) do
            Lecture03.demo(epochs=1)
        end
        @test_nowarn redirect_stdout(devnull) do
            Lecture03.flux_demo(epochs=1)
        end
    end

    @testset "Lecture04" begin
        println("Testing Lecture04...")
        @test_nowarn redirect_stdout(devnull) do
            Lecture04.demo(epochs=1)
        end
        @test_nowarn redirect_stdout(devnull) do
            _, losses = Lecture04.demo(validation_size=1000,epochs=1)
            Lecture04.plot_losses(losses)
        end
    end

    @testset "Lecture05" begin
        println("Testing Lecture05...")
        @test_nowarn redirect_stdout(devnull) do
            Lecture05.filter_image()
        end
        # skip below to prevent failure from network downtime
        # @test_nowarn redirect_stdout(devnull) do
        #    Lecture05.plot_stock_with_moving_average("MSFT", 5, "1mo")
        # end
        @test_nowarn redirect_stdout(devnull) do
            Lecture05.demo(epochs=1)
        end
    end

    @testset "Lecture06" begin
        println("Testing Lecture06...")
        @test_nowarn redirect_stdout(devnull) do
            Lecture06.demo(epochs=1,interactive=false)
        end
    end
end


