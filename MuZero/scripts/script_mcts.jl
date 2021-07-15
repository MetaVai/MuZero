using Base: Float64
using AlphaZero
using ProgressLogging

include("../mu_game_wrapper.jl")
include("../network.jl")
include("../trace.jl")
include("../play.jl")
#include("memory.jl")
include("../training.jl")

# experiment = Examples.experiments["tictactoe"]
# session = Session(experiment)
# env = session.env

# oracle = Network.copy(env.bestnn, on_gpu=env.params.self_play.sim.use_gpu, test_mode=true)
# player = MctsPlayer(env.gspec, oracle, env.params.self_play.mcts)


# game = GI.init(env.gspec)

# function AlphaZero.think(p::MctsPlayer, game)
#     hidden_state = Representation()(GI.current_state(game))
#     mugame = MuGameEnvWrapper(game,
#         CachedOracle(Dynamics(gspec=env.gspec),Dict{Tuple{typeof(hidden_state), Int},Tuple{Float64, typeof(hidden_state)}}()),
#         Representation(),
#         hidden_state,
#         true,
#         GI.white_playing(game),
#         0.)

#     if isnothing(p.timeout) # Fixed number of MCTS simulations
#         MCTS.explore!(p.mcts, mugame, p.niters)
#     else # Run simulations until timeout
#         start = time()
#         while time() - start < p.timeout
#         MCTS.explore!(p.mcts, mugame, p.niters)
#         end
#     end
#     return MCTS.policy(p.mcts, mugame)
# end

# actions, π_target = think(player, game) #overload this function, perhaps MuMCTSplayer

gspec = Examples.games["tictactoe"]

representationhp = RepresentationHP(width=200, depth=4, hiddenstate_shape=32)
representation_oracle = RepresentationNetwork(gspec, representationhp)

dynamicshp = DynamicsHP(hiddenstate_shape=32, width=200, depth_common=4)
dynamics_oracle = DynamicsNetwork(gspec, dynamicshp)

predictionhp = PredictionHP(hiddenstate_shape=32, width=200, depth_common=4)
prediction_oracle = PredictionNetwork(gspec, predictionhp)

nns = (f=prediction_oracle, g=dynamics_oracle, h=representation_oracle)

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

n=10
sim=SimParams(
    num_games=10,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false)

  mcts = MctsParams(
    num_iters_per_turn=400,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0)

  self_play = (; sim, mcts)


  #memory = CircularBuffer{Trace{GI.state_type(gspec)}}(1024)

  μparams = MuParams(self_play, 1, 100)

  env = MuEnv(gspec, μparams, nns)

  t = @timed self_play_step!(env)
  # push!(res, (n, t))

n_workers = 1:16
res2 = []
for n in n_workers
  sim=SimParams(
    num_games=100,
    num_workers=n,
    batch_size=n,
    use_gpu=false,
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false)

  mcts = MctsParams(
    num_iters_per_turn=400,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0)

  self_play = (; sim, mcts)


  #memory = CircularBuffer{Trace{GI.state_type(gspec)}}(1024)

  μparams = MuParams(self_play, 1, 100)

  env = MuEnv(gspec, μparams, nns)

  t = @timed self_play_step!(env)
  push!(res2, (n,t))
  @info res2

end

using Plots
# bar(first.(res), last.(res), xlabel="num_workers", ylabel="time [s]", label="num_games=100")
bar(first.(res2), last.(res2), xlabel="num_workers", ylabel="time [s]", label="num_games=100")
