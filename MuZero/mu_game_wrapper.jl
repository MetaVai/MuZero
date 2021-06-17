import StaticArrays
using Base: @kwdef
#pseudocode 
@kwdef mutable struct Representation
    nn = nothing
end

@kwdef mutable struct Dynamics
    gspec 
    nn = nothing
end

#batchify it
function (representation_h::Representation)(state)
    hidden_state = state 
    return hidden_state
end

#batchify it 
function (dynamics_g::Dynamics)(hidden_state, action)
    game_g = GI.init(dynamics_g.gspec, hidden_state)
    GI.play!(game_g, action)
    reward = GI.white_reward(game_g)
    next_hidden_state = GI.current_state(game_g)
    return (reward, next_hidden_state)
end

# maybe add max_size (?), but it is cleared every think() run
mutable struct CachedOracle{O, K, V}
    oracle :: O
    lookuptable :: Dict{K,V} # Dict{Tuple{typeof(hidden_state), Int},Tuple{Float64, typeof(hidden_state)}}()
end

function (oracle::CachedOracle)(args...)
    if haskey(oracle.lookuptable, args)
        answer = oracle.lookuptable[args]
    else
        answer = oracle.oracle(args...)
        oracle.lookuptable[args] = answer
    end
    return answer
end

#naming: dynamics_oracle, representation_oracle 
# or dynamics_g, representation_h, or just dynamics, representation

mutable struct MuGameEnvWrapper{State, D, R} <: GI.AbstractGameEnv
    game :: GI.AbstractGameEnv
    dynamics_oracle :: D # g(s,a) -> r,s 
    representation_oracle :: R # h(0) -> s
    curstate :: State
    isrootstate :: Bool
    white_playing :: Bool # MuWrapper changes player each round
    lastreward :: Float64
end

GI.game_terminated(game::MuGameEnvWrapper) = false 
# GI.game_terminated(game::MuGameEnvWrapper) = GI.game_terminated(GI.init(GI.spec(game.game),game.curstate))

GI.current_state(game::MuGameEnvWrapper) = game.curstate

GI.available_actions(game::MuGameEnvWrapper) = game.isrootstate ? GI.available_actions(game.game) : GI.actions(GI.spec(game.game))

GI.white_playing(game::MuGameEnvWrapper) = game.white_playing

function GI.play!(game::MuGameEnvWrapper, action)
    # (R, S) = game.dynamics_oracle(game.curstate, action)
    (game.lastreward, game.curstate) = game.dynamics_oracle(game.curstate, action)
    game.white_playing = !game.white_playing
    game.isrootstate = false
end

GI.white_reward(game::MuGameEnvWrapper) = game.lastreward #vulnerable when call white_reward() before play!()

function GI.clone(game::MuGameEnvWrapper) # make lookuptable shallowcopy, or pass it other way
    MuGameEnvWrapper(GI.clone(game.game),
    game.dynamics_oracle,
    game.representation_oracle,
    game.curstate,
    game.isrootstate,
    game.white_playing,
    game.lastreward)
end


