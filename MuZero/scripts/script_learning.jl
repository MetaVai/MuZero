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
include("../simulations.jl")
include("../training.jl")
include("../learning.jl")
include("../benchmark.jl")
include("../probe-games.jl")


device = Flux.cpu
# device = Flux.gpu
# CUDA.allowscalar(true)

gspec = Examples.games["connect-four"]
# gspec = ProbeGames.games["a1r1"]
# gspec = ProbeGames.games["a1o2r2"]
# gspec = ProbeGames.games["a1r1s2"] # representation collapses states into the same one
# gspec = ProbeGames.games["a2r2"] 
# gspec = ProbeGames.games["twoplayer"] 
# gspec = ProbeGames.games["simplertictactoe"] 

n=32
self_play = (;
  sim=SimParams(
    num_games=100,
    num_workers=n,
    # TODO make batch sizes for each network
    batch_size=16, 
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
    dirichlet_noise_α=0.1),
  device=Flux.gpu)

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
    num_workers=16,
    batch_size=16, 
    use_gpu=false,
    reset_every=1,
    flip_probability=0., #0.5
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.0,
  device=Flux.gpu)


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
    num_workers=64,
    batch_size=32,
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
  # μparams = MuParams(self_play, learning_params, arena, 2, 3000)

  # αnns = (
  #   f=PredictionNetwork(
  #     gspec,
  #     PredictionHP(hiddenstate_shape=27, width=256, depth_common=6,
  #       use_batch_norm=true, batch_norm_momentum=1.)),
  #   g=AlphaDynamics(gspec), # set num_unroll_steps=0 to not use it during learning
  #   h=AlphaRepresentation())

  # αnns = (
  #   f=AlphaPrediction(
  #       PredictionNetwork(
  #         gspec,
  #         PredictionHP(hiddenstate_shape=27, width=256, depth_common=6,
  #           use_batch_norm=true, batch_norm_momentum=1.))),
  #   g=AlphaDynamics(gspec), # set num_unroll_steps=0 to not use it during learning
  #   h=AlphaRepresentation())
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
  # env = MuEnv(gspec, μparams, MuNetwork(μNetworkHP), experience=FileIO.load("results/c4_5x4/memory_mctsrollout500.jld2", "mem"))
  env = MuEnv(gspec, μparams, μNetwork, experience=FileIO.load("results/c4_5x4/memory_mctsrollout500.jld2", "mem"))

##

path = "results/c4_5x4/" * format(now(),"yyyy-mm-ddTHHMM") * "_custominit33_testbatching/"
tblogger=TBLogger(path, min_level=Logging.Info)
  # TRAIN
  with_logger(tblogger) do
    @info "params" params=μparams
    # @info "alphalike params" αnns.f.net.hyper
    # @timed learning_step!(env)
    @info "network params" μNetworkHP 
    # @info "Memory Analysis" memory_analysis(env.memory)... "generated by MinMax player"
    @info "Benchmark" run_duel(env, benchmark)...
    mutrain!(env, benchmark=benchmark, path=path)
  end

  # SUPERVISED LEARNING
  with_logger(tblogger) do 
    @info "params" params=μparams
    @info "network params" μNetworkHP 
    # @info "alphalike params" αnns.f.net.hyper
    # train!(env; benchmark=benchmark) 
    @info "Memory Analysis" memory_analysis(env.memory)... "generated by MCTS player"
    @info "Benchmark" run_duel(env, benchmark)...
    function supervised_learning!(env, benchmark, num_iterations, path)
      while env.itc <= num_iterations
        _, time_learning = @timed learning_step!(env)
        @info "Training" stage="iteration finished" env.itc time_learning 
        if env.itc % 50 == 0 
          @info "Benchmark" run_duel(env, benchmark)...
          FileIO.save(path*"$(format(now(),"yyyy-mm-ddTHHMM"))_env_$(env.itc).jld2", "env", env)
        end
        env.itc += 1
      end
    end
    supervised_learning!(env, benchmark, 1500, path)
  end



  with_logger(tblogger) do 
  self_play_step!(env)
end

  @enter self_play_step!(env)
  @profview self_play_step!(env)
  @timed self_play_step!(env)
  @timed learning_step!(env)
  @profview run_duel(env, benchmark)
  mutrain!(env)



  # TRAIN without benchmark
  with_logger(tblogger) do
    @info "params" params=μparams
    # @info "alphalike params" αnns.f.net.hyper
    # @timed learning_step!(env)
    @info "network params" μNetworkHP 
    # @info "Memory Analysis" memory_analysis(env.memory)... "generated by MinMax player"
    mutrain!(env)
  end

  # TRAIN with pretrain
  with_logger(tblogger) do
    @info "params" params=μparams
    # @info "alphalike params" αnns.f.net.hyper
    @info "network params" μNetworkHP 
    # @info "Memory Analysis" memory_analysis(env.memory)... "generated by MinMax player"
    @info "Memory Analysis" memory_analysis(env.memory)... "generated by MinMax player"
    _ = run_duel(env, benchmark)
    for _ in 1:50
      learning_step!(env)
    end
    _ = run_duel(env, benchmark)
    mutrain!(env, benchmark=benchmark)
  end

benchmark = [
  Benchmark.Duel(
    Mu(arena.mcts),
    Benchmark.MctsRollouts(self_play.mcts),
    benchmark_sim)]
benchmark = [
  Benchmark.Duel(
    Mu(arena.mcts),
    Benchmark.MctsRollouts(arena.mcts),
    benchmark_sim)]

run_duel(env,benchmark)
AlphaZero.Util.@printing_errors train!(env, benchmark=benchmark)

self_play_step!(env)
  @enter rewards, redundancy = pit_networks(gspec, env.curnns, env.bestnns, arena)
  rewards, redundancy = pit_networks(gspec, env.curnns, env.bestnns, arena)
  mean(rewards)
# FileIO.save("env_longminmaxlearning.jld2", "env", env)
# FileIO.save("env_muzerogeneral.jld2", "env", env)
# FileIO.save("env_muzerogeneral_minmaxpretrained.jld2", "env", env)
  train!(env)

#generate minmax samples
  # with_logger(tblogger) do
  env = MuEnv(gspec, μparams, αnns)

    @info "params" params=μparams
    AlphaZero.Util.@printing_errors @timed self_play_step!(env)
    @info "Memory Analysis" memory_analysis(env.memory)... "generated by MinMax player τ=0.25, depth=5"
    # FileIO.save("memory_minmax_d5_t025.jld2", "mem", env.memory)
    # FileIO.save("memory_minmax_randominits_d5_t025.jld2", "mem", env.memory)
    # FileIO.save("memory_minmax_d7.jld2", "mem", env.memory)
    # FileIO.save("memory_mcts_rollout_300.jld2", "mem", env.memory)
    # FileIO.save("memory_mctsrollout300_custominit1.jld2", "mem", env.memory)
    # FileIO.save("memory_mctsrollout64.jld2", "mem", env.memory)
    FileIO.save("results/c4_5x4/memory_mctsrollout500.jld2", "mem", env.memory)
  # end

count(last(t.rewards)==(1) for t in env.memory)
env = envres
  hyper = env.params.learning_params
  traces = [sample_trace(env.memory) for _ in 1:2] #? change to iterator
  trace_pos_idxs = [sample_position(t) for t in traces]
  sample = make_target(gspec, traces[1], 2, hyper)
  samples = [make_target(gspec, t, i, hyper) for (t,i) in zip(traces, trace_pos_idxs)]
  # FileIO.save("sample.jld2", "sample_st6",sample)
  sample = FileIO.load("sample.jld2", "sample_st6")
samples = [sample]
  X       = Flux.batch(smpl.x       for smpl in samples)
  A_mask  = Flux.batch(smpl.a_mask  for smpl in samples)
  As      = Flux.batch(smpl.as      for smpl in samples)
  Ps      = Flux.batch(smpl.ps      for smpl in samples)
  Vs      = Flux.batch(smpl.vs      for smpl in samples)
  Rs      = Flux.batch(smpl.rs      for smpl in samples)
  f32(arr) = convert(AbstractArray{Float32}, arr)
data =  map(x -> f32(x), (; X, A_mask, As, Ps, Vs, Rs))
f32(As)
(data.X, data.As)


  # data = [sample_batch(gspec, env.memory, env.params.learning_params) for _ in 1:1]
  losses(env.curnns, env.params.learning_params, data)
  @enter losses(env.curnns, env.params.learning_params, data)

  @enter sample_batch(gspec, envres.memory, envres.params.learning_params)
  samples = (sample_batch(gspec, envres.memory, envres.params.learning_params)|>gpu for _ in 1:n)
  for spl in samples
    @info spl
  end

  t = @timed self_play_step!(env)
  @timed learning_step!(env)
  # @enter self_play_step!(env)
  @enter learning_step!(env)

  # FileIO.save("env.jld2", env)
  mem_test = FileIO.load("memory_minmax.jld2", "mem")

  # for (b1,b2) in zip(env.memory, mem_test)
  #   @assert b1 ≂ b2
  # end
  # mem_test["mem"].buffer  env.memory.buffer
  # mem_test["mem"].buffer[1]
  # env.memory.buffer[1]       

  # learning_step
lp = env.params.learning_params
tr = MuTrainer(env.gspec, env.curnns|>lp.device|>Flux.trainmode!, env.memory, env.params.learning_params, env.params.learning_params.opt)
nbatches = lp.batches_per_checkpoint
env.curnns|>Flux.gpu
update_weights!(tr, nbatches)
@timed samples = [sample_batch(tr.gspec, tr.memory, tr.hyper) for _ in 1:10] |> Flux.gpu

counter=0
for d in data
  counter += 1
end
counter
d = samples[1]
losses(tr.nns, tr.hyper, d)
@info d.X[:,:,1,1]




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