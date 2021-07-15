using AlphaZero
using ProgressLogging, TensorBoardLogger, Logging
import Flux

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include("../mu_game_wrapper.jl")
include("../network.jl")
include("../trace.jl")
include("../play.jl")
include("../training.jl")
include("../learning.jl")
include("../benchmark.jl")

tblogger=TBLogger("tensorboard_logs/run", min_level=Logging.Info)

gspec = Examples.games["tictactoe"]

μNetworkHP = MuNetworkHP(gspec,
  PredictionHP(hiddenstate_shape=32, width=256, depth_common=4),
  DynamicsHP(hiddenstate_shape=32, width=256, depth_common=4),
  RepresentationHP(width=256, depth=2, hiddenstate_shape=32))
μNetworkHP = MuNetworkHP(gspec,
  PredictionHP(hiddenstate_shape=32, width=64, depth_common=4),
  DynamicsHP(hiddenstate_shape=32, width=64, depth_common=4),
  RepresentationHP(width=64, depth=4, hiddenstate_shape=32))

n=1
self_play = (;
  sim=SimParams(
    num_games=512,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    reset_every=4, #not used, mcts resets everytime
    flip_probability=0.,
    alternate_colors=false),
  mcts = MctsParams(
    num_iters_per_turn=64,
    cpuct=2.5,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.5))

arena = (;
  sim=SimParams(
    num_games=100,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.5,
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.00)


learning_params = (;
  num_unroll_steps=5,
  td_steps=9,
  discount=0.997,
  l2_regularization=1e-4,
  loss_computation_batch_size=64,
  batches_per_checkpoint=512,
  num_checkpoints=1,
  learning_rate=0.003,
  momentum=0.9)

benchmark_sim = SimParams(
    num_games=100,
    num_workers=4,
    batch_size=4,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.5,
    alternate_colors=true)

benchmark = [
  Benchmark.Duel(
    Mu(self_play.mcts),
    Benchmark.MctsRollouts(self_play.mcts),
    benchmark_sim)]

  #memory = CircularBuffer{Trace{GI.state_type(gspec)}}(1024)

  μparams = MuParams(self_play, learning_params, arena, 40, 3000)

  env = MuEnv(gspec, μparams, MuNetwork(μNetworkHP))
  with_logger(tblogger) do 
    @info "params" params=μparams
    train!(env; benchmark=benchmark) 
  end
  train!(env)

  t = @timed self_play_step!(env)
  @timed learning_step!(env)
  @enter self_play_step!(env)

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
# nns = deepcopy(env.bestnns)
# nns ≂ env.bestnns
# fieldnames(typeof(nns))
# typeof(nns) == typeof(env.bestnns)
# nns.f.common

(X, A_mask, As, Ps, Vs, Rs) = sample_batch(env.gspec, env.memory, env.params.learning_params)
losses(env.curnns, learning_params, (X, A_mask, As, Ps, Vs, Rs))
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