using Base: Float32
using AlphaZero

include("mu_game_wrapper.jl")
include("network.jl")

experiment = Examples.experiments["tictactoe"]
session = Session(experiment)
env = session.env

oracle = Network.copy(env.bestnn, on_gpu=env.params.self_play.sim.use_gpu, test_mode=true)
player = MctsPlayer(env.gspec, oracle, env.params.self_play.mcts)


game = GI.init(env.gspec)

function AlphaZero.think(p::MctsPlayer, game)
    hidden_state = Representation()(GI.current_state(game))
    mugame = MuGameEnvWrapper(game,
        CachedOracle(Dynamics(gspec=env.gspec),Dict{Tuple{typeof(hidden_state), Int},Tuple{Float64, typeof(hidden_state)}}()),
        Representation(),
        hidden_state, 
        true,
        GI.white_playing(game),
        0.)

    if isnothing(p.timeout) # Fixed number of MCTS simulations
        MCTS.explore!(p.mcts, mugame, p.niters)
    else # Run simulations until timeout
        start = time()
        while time() - start < p.timeout
        MCTS.explore!(p.mcts, mugame, p.niters)
        end
    end
    return MCTS.policy(p.mcts, mugame)
end

actions, π_target = think(player, game) #overload this function, perhaps MuMCTSplayer

gspec = env.gspec

representationHP = RepresentationHP(width=200,depth=4,hiddenstate_shape=32)
representation_oracle = RepresentationNetwork(env.gspec, representationHP)

dynamicshp = DynamicsHP(hiddenstate_shape=32,width=200,depth_common=4)
dynamics_oracle = DynamicsNetwork(gspec, dynamicshp)

predictionhp = PredictionHP(hiddenstate_shape=32,width=200,depth_common=4)
prediction_oracle = PredictionNetwork(gspec, predictionhp)


function AlphaZero.MctsPlayer(
    game_spec::AbstractGameSpec, oracle, params::MctsParams; timeout=nothing, S=GI.state_type(game_spec))
  mcts = MCTS.Env(game_spec, oracle,
    gamma=params.gamma,
    cpuct=params.cpuct,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α,
    prior_temperature=params.prior_temperature,
    S=S)
  return MctsPlayer(mcts,
    niters=params.num_iters_per_turn,
    τ=params.temperature,
    timeout=timeout)
end



player = MctsPlayer(env.gspec, prediction_oracle, env.params.self_play.mcts; S=Vector{Float32})

function AlphaZero.think(p::MctsPlayer, game, representation_oracle, dynamics_oracle)
    hidden_state = representation_oracle(GI.current_state(game))
    mugame = MuGameEnvWrapper(game,
        CachedOracle(dynamics_oracle,Dict{Tuple{typeof(hidden_state), Int},Tuple{Float64, typeof(hidden_state)}}()),
        representation_oracle,
        hidden_state, 
        true,
        GI.white_playing(game),
        0.)

    if isnothing(p.timeout) # Fixed number of MCTS simulations
        MCTS.explore!(p.mcts, mugame, p.niters)
    else # Run simulations until timeout
        start = time()
        while time() - start < p.timeout
        MCTS.explore!(p.mcts, mugame, p.niters)
        end
    end
    return MCTS.policy(p.mcts, mugame)
end

@enter actions, π_target = think(player, game, representation_oracle, dynamics_oracle)