using Revise
using AlphaZero
using ProgressLogging
using TensorBoardLogger
using Logging
using ParameterSchedulers: Scheduler, Cos
import Flux
import FileIO
import Random
import CUDA

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include("../mu_game_wrapper.jl")
# include("../architectures/simplemlp.jl")
# include("../architectures/resnet.jl")
include("../network.jl")

# include("../alphazerolike.jl")
include("../trace.jl")
include("../play.jl")
include("../simulations.jl")
include("../training.jl")
include("../learning.jl")
include("../benchmark.jl")
include("../probe-games.jl")

# 3x Faster than julia default implementation #! may break something
Base.hash(x::Array{Float32,3}) = hash(vec(x))
Base.hash(x::Tuple{Array{Float32,3},Int}) = hash(x[1]) >> x[2]


device = Flux.cpu
# device = Flux.gpu
# CUDA.allowscalar(false)

gspec = Examples.games["connect-four"]



##


n=1
self_play = (;
  sim=SimParams(
    num_games=100,
    num_workers=1,
    # TODO make batch sizes for each network
    batch_size=1, 
    use_gpu=true,
    # todevice=Flux.cpu, #? 
    reset_every=4, #not used, mcts resets everytime
    flip_probability=0.,
    alternate_colors=false),
  mcts = MctsParams(
    num_iters_per_turn=128, #1000 benchmark
    cpuct=1.25,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.1),
  device=Flux.gpu)

arena = (;
  sim=SimParams(
    num_games=4,
    num_workers=4,
    batch_size=4, 
    use_gpu=true,
    reset_every=1,
    flip_probability=0., #0.5
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.4,
  device=Flux.gpu)


learning_params = (;
  num_unroll_steps=10, #if =0, g is not learning, muzero-general=20
  td_steps=50, # with max length=9, always go till the end of the game, rootvalues don't count
  discount=0.997,
  #// value_loss_weight = 0.25, #TODO
  l2_regularization=1f-4, #Float32
  #// l2_regularization=0f0, #Float32
  loss_computation_batch_size=128,
  batches_per_checkpoint=10,
  num_checkpoints=1,
  opt=Scheduler(
    Cos(λ0=1e-2, λ1=1e-5, period=10000), # cosine annealing, google 2e4, generat doesn't use any
    Flux.ADAM()
  ),
  device=Flux.gpu
)

benchmark_sim = SimParams(
    num_games=400,
    num_workers=64,
    batch_size=32,
    use_gpu=true,
    reset_every=1,
    flip_probability=0.0, #0.5 
    alternate_colors=true)

bench_mcts = MctsParams(
  num_iters_per_turn=500, #1000 benchmark
  cpuct=1.25,
  temperature=ConstSchedule(0.3),
  dirichlet_noise_ϵ=0.25,
  dirichlet_noise_α=0.1)

benchmark = (;
  vanilla_mcts=Benchmark.Duel(
    Mu(arena.mcts),
    Benchmark.MctsRollouts(bench_mcts),
    benchmark_sim),
  # minmax_d5=Benchmark.Duel(
  #   Mu(arena.mcts),
  #   Benchmark.MinMaxTS(depth=5, amplify_rewards=true, τ=1.),
  #   benchmark_sim)
)


  μparams = MuParams(self_play, learning_params, arena, 1_000, 10_000)

# muzero-general params
simμNetworkHP = MuNetworkHP(
  gspec,
  PredictionSimpleHP(hiddenstate_shape=64, width=64, depth_common=0,
    depth_vectorhead=0, depth_scalarhead=0, use_batch_norm=false, batch_norm_momentum=1.),
  DynamicsSimpleHP(hiddenstate_shape=64, width=64, depth_common=-1,
   depth_vectorhead=0, depth_scalarhead=0,  use_batch_norm=false, batch_norm_momentum=1.),
  RepresentationSimpleHP(hiddenstate_shape=64, width=0, depth=-1))

  num_filters=32
resμNetworkHP = MuNetworkHP(
  gspec,
  PredictionResnetHP(
    num_blocks=3,
    num_filters=num_filters,
    conv_kernel_size=(3,3),
    num_policy_head1_filters=32,
    num_value_head2_filters=32,
    batch_norm_momentum=0.6f0
  ), 
  DynamicsResnetHP(
    num_blocks=3,
    num_filters=num_filters,
    conv_kernel_size=(3,3),
    num_reward_head2_filters=32,
    batch_norm_momentum=0.6f0
  ),
  RepresentationResnetHP(
    num_blocks=3,
    num_filters=num_filters,
    conv_kernel_size=(3,3),
    batch_norm_momentum=0.6f0
  )
)

  envsim = MuEnv(gspec, μparams, MuNetwork(simμNetworkHP))
  envres = MuEnv(gspec, μparams, MuNetwork(resμNetworkHP))
  # env = MuEnv(gspec, μparams, deepcopy(env.bestnns))
  # env = MuEnv(gspec, μparams, MuNetwork(μNetworkHP), experience=FileIO.load("results/c4_5x4/memory_mctsrollout500.jld2", "mem"))
  # env = MuEnv(gspec, μparams, μNetwork, experience=FileIO.load("results/c4_5x4/memory_mctsrollout500.jld2", "mem"))
##

# @profview self_play_step!(env)




# # simple test
# game = GI.init(gspec)

# GI.play!(game, 3)
# # hiddenstate = h(game)
# onehotaction = GI.encode_action(gspec,a)
# cat(hiddenstate, onehotaction, dims=3)

# state1 = GI.current_state(game)
# GI.play!(game, 3)
# state2 = GI.current_state(game)
# GI.play!(game, 3)
# state3 = GI.current_state(game)

# states = [state1, state2, state3]

# vstates = Flux.batch(GI.vectorize_state(gspec, s) for s in states)
# hiddenstates = forward(h, vstates)

# actions = [3,3,1]
# onehotactions = Flux.batch(GI.encode_action(gspec, a) for a in actions)

# g((hiddenstates[:,:,:,1],3))
# hiddenstates[:,:,:,1]
# stateactions = cat(hiddenstates, onehotactions, dims=3)
# forward(g, stateactions)

# forward(f, hiddenstates)


