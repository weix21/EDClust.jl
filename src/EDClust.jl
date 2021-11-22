module EDClust

using Distributions
using SpecialFunctions
using StatsBase
using ProgressMeter

include("EMPolya.jl")
include("MMPolya.jl")
include("fitPolya.jl")

export fitPolya

end
