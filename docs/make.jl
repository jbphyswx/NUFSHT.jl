using Documenter: Documenter
using NUFSHT: NUFSHT

Documenter.makedocs(;
    modules  = [NUFSHT],
    sitename = "NUFSHT.jl",
    authors  = "jbphyswx",
    format   = Documenter.HTML(;
        prettyurls  = get(ENV, "CI", "false") == "true",
        canonical   = "https://jbphyswx.github.io/NUFSHT.jl",
        edit_link   = "main",
    ),
    pages = [
        "Home"      => "index.md",
        "Algorithm" => "algorithm.md",
        "API"       => "api.md",
    ],
    warnonly = false,
)

Documenter.deploydocs(;
    repo   = "github.com/jbphyswx/NUFSHT.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
