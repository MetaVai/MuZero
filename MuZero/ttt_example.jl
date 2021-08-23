module TTTexample
import Flux

using AlphaZero
using ..MuZero

using ParameterSchedulers: Scheduler, Cos


gspec = Examples.games["tictactoe"]

n=16
self_play = (;
  sim=SimParams(
    num_games=100,
    num_workers=n,
    batch_size=8,
    use_gpu=false, # not used, change device to Flux.gpu
    # todevice=Flux.cpu, #? 
    reset_every=4, #not used, mcts resets everytime
    flip_probability=0.,
    alternate_colors=false),
  mcts = MctsParams(
    num_iters_per_turn=64, #1000 benchmark
    cpuct=1.25,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.1),
    device=Flux.cpu) # Flux.cpu || Flux.gpu

arena = (;
  sim=SimParams(
    num_games=50,
    num_workers=n,
    batch_size=n÷2,
    use_gpu=false, # not used, change device to Flux.gpu
    reset_every=1,
    flip_probability=0., #0.5
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.0,
  device=Flux.cpu)# Flux.cpu || Flux.gpu


learning_params = (;
  num_unroll_steps=5, #if =0, g is not learning, muzero-general=20
  td_steps=20, # with max length=9, always go till the end of the game, rootvalues don't count
  discount=0.997,
  #// value_loss_weight = 0.25, #TODO
  l2_regularization=1f-4, #Float32
  #// l2_regularization=0f0, #Float32
  loss_computation_batch_size=512,
  batches_per_checkpoint=10,
  num_checkpoints=1,
  opt=Scheduler(
    Cos(λ0=3e-3, λ1=1e-5, period=10000), # cosine annealing, google period 2e4, generat doesn't use any
    Flux.ADAM()
  ),
  model_type = :mlp, # :mlp || :resnet
  device=Flux.cpu # Flux.cpu || Flux.gpu
)

benchmark_sim = SimParams(
    num_games=400,
    num_workers=n,
    batch_size=n÷2,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.0, #0.5 
    alternate_colors=true)

bench_mcts = MctsParams(
  num_iters_per_turn=400, #1000 benchmark
  cpuct=1.25,
  temperature=ConstSchedule(0.3),
  dirichlet_noise_ϵ=0.25,
  dirichlet_noise_α=0.1)

benchmark = (;
  vanilla_mcts=Benchmark.Duel(
    Mu(arena.mcts),
    Benchmark.MctsRollouts(bench_mcts),
    benchmark_sim),
  minmax_d5=Benchmark.Duel(
    Mu(arena.mcts),
    Benchmark.MinMaxTS(depth=5, amplify_rewards=true, τ=1.),
    benchmark_sim)
)


  μparams = MuParams(self_play, learning_params, arena, 5_000, 3000)

# muzero-general params
# μNetworkHP = MuNetworkHP(
#   gspec,
#   PredictionHP(hiddenstate_shape=32, width=64, depth_common=-1,
#     depth_vectorhead=0, depth_scalarhead=0, use_batch_norm=false, batch_norm_momentum=1.),
#   DynamicsHP(hiddenstate_shape=32, width=64, depth_common=-1,
#    depth_vectorhead=0, depth_scalarhead=0s,  use_batch_norm=false, batch_norm_momentum=1.),
#   RepresentationHP(hiddenstate_shape=32, width=0, depth=-1))

  hs_shape = 32
  simμNetworkHP = MuNetworkHP(
    gspec,
    PredictionSimpleHP(hiddenstate_shape=hs_shape, width=64, depth_common=-1,
      depth_vectorhead=0, depth_scalarhead=0, use_batch_norm=false, batch_norm_momentum=0.8f0),
    DynamicsSimpleHP(hiddenstate_shape=hs_shape, width=64, depth_common=-1,
     depth_vectorhead=0, depth_scalarhead=0,  use_batch_norm=false, batch_norm_momentum=0.8f0),
    RepresentationSimpleHP(hiddenstate_shape=hs_shape, width=0, depth=-1))

  env = MuEnv(gspec, μparams, MuNetwork(simμNetworkHP))

end # module
