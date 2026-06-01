using Documenter
using NUFSHT
using FastSphericalHarmonics

makedocs(;
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
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(;
    repo   = "github.com/jbphyswx/NUFSHT.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
