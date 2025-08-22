using Documenter
using Pkg

# Activate the parent project to access MachineLearningCourse
Pkg.activate("..")
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
        "Lecture 03: Deep Neural Networks" => "lecture03.md"
    ],
    checkdocs = :exports,
    doctest = false,  # Skip doctests for now
)

# Deploy docs only if running on CI
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/rajgoel/course-machine-learning.git",
        target = "build",
        branch = "gh-pages",
        devbranch = "main",
        dirname = "julia",  # This puts docs in julia/ subdirectory
    )
end
