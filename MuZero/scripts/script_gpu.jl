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
include("../network.jl")
include("../alphazerolike.jl")
include("../trace.jl")
include("../play.jl")
include("../training.jl")
include("../learning.jl")
include("../benchmark.jl")
include("../probe-games.jl")

device = Flux.cpu
# device = Flux.gpu
CUDA.allowscalar(true)

gspec = Examples.games["connect-four"]
# gspec = ProbeGames.games["a1r1"]
# gspec = ProbeGames.games["a1o2r2"]
# gspec = ProbeGames.games["a1r1s2"] # representation collapses states into the same one
# gspec = ProbeGames.games["a2r2"] 
# gspec = ProbeGames.games["twoplayer"] 
# gspec = ProbeGames.games["simplertictactoe"] 

n=8
self_play = (;
  sim=SimParams(
    num_games=5,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    # todevice=Flux.cpu, #? 
    reset_every=4, #not used, mcts resets everytime
    flip_probability=0.,
    alternate_colors=false),
  mcts = MctsParams(
    num_iters_per_turn=128, #1000 benchmark
    cpuct=1.25,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.1))

# # for mcts rollout generation
# self_play = (;
#   sim=SimParams(
#     num_games=10_000,
#     num_workers=n,
#     batch_size=n,
#     use_gpu=false,
#     reset_every=4, #not used, mcts resets everytime
#     flip_probability=0.,
#     alternate_colors=false),
#   mcts = MctsParams(
#     num_iters_per_turn=500, #1000 benchmark
#     cpuct=2.5,
#     temperature=ConstSchedule(1.0),
#     dirichlet_noise_ϵ=0.25,
#     dirichlet_noise_α=0.1))

arena = (;
  sim=SimParams(
    num_games=50,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    reset_every=1,
    flip_probability=0., #0.5
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.0,
  device=Flux.cpu)


learning_params = (;
  num_unroll_steps=10, #if =0, g is not learning, muzero-general=20
  td_steps=50, # with max length=9, always go till the end of the game, rootvalues don't count
  discount=0.997,
  #// value_loss_weight = 0.25, #TODO
  l2_regularization=1f-4, #Float32
  #// l2_regularization=0f0, #Float32
  loss_computation_batch_size=512,
  batches_per_checkpoint=10,
  num_checkpoints=1,
  opt=Scheduler(
    Cos(λ0=1e-3, λ1=1e-5, period=10000), # cosine annealing, google 2e4, generat doesn't use any
    Flux.ADAM()
  ),
  device=Flux.gpu
)

benchmark_sim = SimParams(
    num_games=400,
    num_workers=1,
    batch_size=4,
    use_gpu=false,
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

  #memory = CircularBuffer{Trace{GI.state_type(gspec)}}(1024)

  # μparams = MuParams(self_play, learning_params, arena, 6, 3000)
  μparams = MuParams(self_play, learning_params, arena, 10_000, 10_000)

  Random.seed!(2137)
  # env = MuEnv(gspec, μparams, αnns, experience=FileIO.load("memory_minmax.jld2", "mem"))
  # env = MuEnv(gspec, μparams, αnns, experience=FileIO.load("memory_minmax_d5_t025.jld2", "mem"))


# muzero-general params
μNetworkHP = MuNetworkHP(
  gspec,
  PredictionHP(hiddenstate_shape=64, width=64, depth_common=0,
    depth_vectorhead=0, depth_scalarhead=0, use_batch_norm=false, batch_norm_momentum=1.),
  DynamicsHP(hiddenstate_shape=64, width=64, depth_common=-1,
   depth_vectorhead=0, depth_scalarhead=0,  use_batch_norm=false, batch_norm_momentum=1.),
  RepresentationHP(hiddenstate_shape=64, width=0, depth=-1))

  # env = MuEnv(gspec, μparams, MuNetwork(μNetworkHP))
  # env = MuEnv(gspec, μparams, deepcopy(env.bestnns))
  env = MuEnv(gspec, μparams, MuNetwork(μNetworkHP), experience=FileIO.load("results/c4_5x4/memory_mctsrollout500.jld2", "mem"))

##
learning_step!(env)
##

  # learning_step
lp = env.params.learning_params
tr = MuTrainer(env.gspec, env.curnns|>lp.device|>Flux.trainmode!, env.memory, env.params.learning_params, env.params.learning_params.opt)
@timed samples = [sample_batch(tr.gspec, tr.memory, tr.hyper) for _ in 1:10] |> Flux.gpu

hyper = tr.hyper
nns = env.curnns|>Flux.gpu
(X, A_mask, As, Ps, Vs, Rs) = samples[1]
##
# function losses(nns, hyper, (X, A_mask, As, Ps, Vs, Rs))
  prediction, dynamics, representation = nns.f, nns.g, nns.h
  creg = hyper.l2_regularization
  Ksteps = hyper.num_unroll_steps

  # initial step, from the real observation
  Hiddenstate = forward(representation, X)
  P̂⁰, V̂⁰ = forward(prediction, Hiddenstate)
  P̂⁰ = normalize_p(P̂⁰, A_mask)
  # R̂⁰ = zero(V̂⁰)

  scale_initial = iszero(Ksteps) ? 1f0 : 0.5f0
  Lp = scale_initial * lossₚ(P̂⁰, Ps[:, 1, :]) # scale=1
  Lv = scale_initial * lossᵥ(V̂⁰, Vs[1:1, :])
  Lr = zero(Lv) # starts at next step (see MuZero paper appendix)
  
  scale_recurrent = iszero(Ksteps) ? nothing : 0.5f0 / Ksteps #? instead of constant scale, maybe 2^(-i+1)
  # recurrent inference 
  # for k in 1:Ksteps
  k=1
    # targets are stored as follows: [A⁰¹ A¹² ...] [P⁰ P¹ ...] [V⁰ V¹ ...] but [R¹ R² ...]
    A = As[k, :]
##
# function forward(nn::DynamicsNetwork, hiddenstate, action)
  nn=dynamics; hiddenstate=Hiddenstate; action=A;
  action_one_hot = onehotbatch(action, GI.actions(gspec)) #! scalar indexing
  hiddenstate_action = vcat(hiddenstate, collect(action_one_hot)) #!CUDA ERROR
  c = nn.common(hiddenstate_action)
  r = nn.scalarhead(c)
  s⁺¹ = nn.vectorhead(c)
  return (r, s⁺¹)
# end

    R̂, Hiddenstate = forward(dynamics, Hiddenstate, A) #
    P̂, V̂ = forward(prediction, Hiddenstate) #? should flip V based on players
    # scale loss so that the overall weighting of the recurrent_inference (g,f nns)
    # is equal to that of the initial_inference (h,f nns)
    Lp += scale_recurrent * lossₚ(P̂, Ps[:, k+1, :]) #? @view
    Lv += scale_recurrent * lossᵥ(V̂, Vs[k+1:k+1, :])
    Lr += scale_recurrent * lossᵣ(R̂, Rs[k:k, :])
  # end
  Lreg = iszero(creg) ? zero(Lv) : creg * sum(sum(w.^2) for w in regularized_params(nns))
  L = Lp + Lv + Lr + Lreg # + Lr
  # L = Lp + Lreg # + Lr
  Zygote.@ignore @info "Loss" loss_total=L loss_policy=Lp loss_value=Lv loss_reward=Lr loss_reg_params=Lreg relative_entropy=Lp-Flux.Losses.crossentropy(Ps, Ps) #? check if compute means inside logger is avaliable
  return (L, Lp, Lv, Lr, Lreg)
# end
##

#TODO check with MCTS, MinMax
#Revise.jl, tensorboard_logger

# # checking Losses
# zero(Ps)
# one.(Ps)
# Flux.Losses.mse(zero(Vs), one.(Vs), agg=x->mean(∇_scale .* x)/sum(∇_scale)) 
# Flux.Losses.mse(zero(Vs), one.(Vs), agg=x->mean(x .* ∇_scale)) 
# Flux.Losses.mse(zero(Vs), one.(Vs), agg=x->sum(∇_scale .* x) / sum(∇_scale)) 
# Flux.Losses.mse(zero(Vs), one.(Vs)) 
# sum(∇_scale)
# one.(Vs)
# Vs
# one.(Ps) .* ∇_scale

# Flux.Losses.crossentropy(one.(Ps).*ℯ, one.(Ps), agg=x->mean(x,∇_scale))
# log.(one.(Ps) .* ℯ)
# log.(zero(Ps)+eps.(Ps))
# one.(Ps) .* 
# one.(Ps) .* log.(one.(Ps).*ℯ)
# mean(.-sum(one.(Ps) .* log.(one.(Ps).*ℯ), dims=1), weights(∇_scale))
# ∇_scale .* 2

"Lenient comparison operator for `struct`, both mutable and immutable (type with \\eqsim)."
@generated function ≂(x, y)
  if !isempty(fieldnames(x)) && x == y
    mapreduce(n -> :(x.$n ≂ y.$n), (a,b)->:($a && $b), fieldnames(x))
  else
    :(x == y)
  end
end

# nns = deepcopy(tr.nns)
# nns == tr.nns
# nns ≂ tr.nns
# env.curnns ≂ env.bestnns
# nns = deepcopy(env.bestnns)
# nns ≂ env.bestnns
# fieldnames(typeof(nns))
# typeof(nns) == typeof(env.bestnns)
# nns.f.common

(X, A_mask, As, Ps, Vs, Rs) = sample_batch(env.gspec, env.memory, env.params.learning_params)
@enter losses(env.curnns, learning_params, (X, A_mask, As, Ps, Vs, Rs))
As
A = As[1:1,:]
Vs


env.curnns = deepcopy(env.bestnns)
@assert env.curnns ≂ env.bestnns    # all their fields are the same
@assert !(env.curnns == env.bestnns) # they have not the same memory adress

tr = MuTrainer(env.gspec, env.curnns, env.memory, env.params.learning_params, Flux.ADAM())
@assert env.curnns ≂ tr.nns 
@assert env.curnns == tr.nns # same memory adress (shallow copy)


@timed update_weights!(tr, tr.hyper.batches_per_checkpoint)
@assert env.curnns == tr.nns
@assert !(env.bestnns ≂ tr.nns) # after update! tr.nns and env.curnns changed
@assert !(env.curnns ≂ env.bestnns)


learning_step!(env)
train!(env)


# gnerator with rand
function foo(x)
  y = x+1
  z = x+1
  return (; x,y,z)
end
rands = (rand() for _ in 1:5)
foos = (foo(r) for r in rands)
X = [f.x for f in foos] # Flux.batch(f.x for f in foos)
Y = [f.y for f in foos]
Z = [f.z for f in foos]
X[1], Y[1], Z[1]