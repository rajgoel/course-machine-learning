using Documenter
using Pkg

# Include the module source directly (no package loading)
include("../src/MachineLearningCourse.jl")
include("../03-lecture/src/Lecture03.jl")

# Use the modules directly
using .MachineLearningCourse

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
