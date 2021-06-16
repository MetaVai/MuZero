using AlphaZero

include("mu_game_wrapper.jl")

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