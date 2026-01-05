# --- disable SDL audio for docs ---
ENV["SDL_AUDIODRIVER"] = "dummy"          # SDL uses a dummy driver
ENV["ALSA_CONFIG_PATH"] = "/dev/null"     # neuter ALSA
ENV["JULIA_SDL2_DISABLE_AUDIO"] = "1"     # SDL2_jll respects this
ENV["SDL_VIDEODRIVER"] = "dummy"          # optional, prevents video init if needed
ENV["DOCUMENTER"] = "true"                # Documenter-specific flag

# --- activate docs environment and dev-add main package ---
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using Documenter
using MachineLearningCourse

makedocs(
    sitename = "MachineLearningCourse.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://rajgoel.github.io/course-machine-learning/julia/",
        assets = String[],
    ),
    modules = [MachineLearningCourse],
    authors = "Asvin Goel",
    pages = [
        "Home" => "index.md",
        "Lectures" => [
            "Lecture 01 - Linear Classifier" => "lecture01.md",
            "Lecture 02 - Gradient Descent" => "lecture02.md", 
            "Lecture 03 - Deep Neural Networks" => "lecture03.md",
            "Lecture 04 - Stochastic Gradient Descent" => "lecture04.md",
            "Lecture 05 - Filtering, Pooling, and Convolution" => "lecture05.md",
            "Lecture 06 - Graph Convolutional Networks" => "lecture06.md",
            "Lecture 07 - Autoencoders" => "lecture07.md",
            "Lecture 08 - Deep Q-Learning" => "lecture08.md",
            "Lecture 09 - Policy Gradients" => "lecture09.md"
        ]
    ],
    checkdocs = :none,
    doctest = false,
    linkcheck = false,
)

# Docs are built to julia/docs/build/ and will be published by GitHub's automatic workflow.

