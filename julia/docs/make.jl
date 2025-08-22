using Documenter
using Pkg

# Activate the parent project to access MachineLearningCourse
Pkg.activate("..")
using MachineLearningCourse

makedocs(
    sitename = "MachineLearningCourse.jl",
    format = Documenter.HTML(
        prettyurls = false,  # For local viewing
        canonical = "",
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
