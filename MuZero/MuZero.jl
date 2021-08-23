module MuZero

export Mu, MuParams, MuNetworkHP,
  PredictionSimpleHP, DynamicsSimpleHP, RepresentationSimpleHP,
  PredictionResnetHP, DynamicsResnetHP, RepresentationResnetHP,
  MuEnv, mutrain!, self_play_step!, learning_step!, MuNetwork,
  run_duel

# 3x Faster than julia 1.6.2 default implementation #! may break something
Base.hash(x::Array{Float32,3}) = hash(vec(x))
Base.hash(x::Tuple{Array{Float32,3},Int}) = hash(x[1]) >> x[2]

using AlphaZero
using ProgressLogging
using TensorBoardLogger
using Logging
using ParameterSchedulers: Scheduler, Cos
import Flux
import FileIO
import CUDA

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include("mu_game_wrapper.jl")
include("network.jl")
# include("../alphazerolike.jl")
include("trace.jl")
include("play.jl")
include("simulations.jl")
include("training.jl")
include("learning.jl")
include("benchmark.jl")
include("probe-games.jl")

include("ttt_example.jl")
import .TTTexample

end # module