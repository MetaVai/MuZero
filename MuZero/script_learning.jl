using AlphaZero
using ProgressLogging
import Flux

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include("mu_game_wrapper.jl")
include("network.jl")
include("trace.jl")
include("play.jl")
include("training.jl")
include("learning.jl")

gspec = Examples.games["tictactoe"]

μNetworkHP = MuNetworkHP(gspec,
  PredictionHP(hiddenstate_shape=32, width=200, depth_common=4),
  DynamicsHP(hiddenstate_shape=32, width=200, depth_common=4),
  RepresentationHP(width=200, depth=4, hiddenstate_shape=32))

n=8
sim=SimParams(
    num_games=50,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    reset_every=4, #not used, mcts resets everytime
    flip_probability=0.,
    alternate_colors=false)

  mcts_params = MctsParams(
    num_iters_per_turn=400,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0)

  self_play_params = (; sim, mcts_params)

learning_params = (;
  num_unroll_steps=4,
  td_steps=5,
  discount=1,
  l2_regularization=1e-4,
  loss_computation_batch_size=32,
  batches_per_checkpoint = 50,
  num_checkpoints=4,
)

  #memory = CircularBuffer{Trace{GI.state_type(gspec)}}(1024)

  μparams = MuParams(self_play_params, learning_params, 2, 1024)

  env = MuEnv(gspec, μparams, MuNetwork(μNetworkHP))
  train!(env)

  t = @timed self_play_step!(env)


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