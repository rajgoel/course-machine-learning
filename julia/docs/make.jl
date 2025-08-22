using Documenter
using Pkg

# Add the parent project to the load path for docstring extraction
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Import the module without precompiling heavy dependencies
import MachineLearningCourse

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
        "Lecture 03: Deep Neural Networks" => "lecture03.md"
    ],
    checkdocs = :exports,
    doctest = false,  # Skip doctests for now
)

# Docs are built to julia/docs/build/ and will be published by GitHub's automatic workflow
# No manual deployment needed - GitHub Pages will serve the entire repository
