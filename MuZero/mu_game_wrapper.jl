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

#batchified
function (representation_h::Representation)(state)
    hidden_state = state 
    return hidden_state
end

#batchified
function (dynamics_g::Dynamics)(hidden_state, action)
    game_g = GI.init(dynamics_g.gspec, hidden_state)
    GI.play!(game_g, action)
    reward = GI.white_reward(game_g)
    next_hidden_state = GI.current_state(game_g)
    return (reward, next_hidden_state)
end


#naming: dynamics_oracle, representation_oracle 
# or dynamics_g, representation_h

mutable struct MuGameEnvWrapper{G, S, A} <: GI.AbstractGameEnv
    game :: G
    dynamics_oracle :: Dynamics # g(s,a) -> r,s 
    representation_oracle :: Representation # h(0) -> s
    curstate :: S
    rootstate :: S # utilize it better (play! should set it to false, and clone to reset)
    white_playing :: Bool
    lookuptable :: Dict{Tuple{S,A},Tuple{Float64,S}} # (state,action) => (reward, state) #maybe structs? or named tuples
    lastreward :: Float64
end

# GI.game_terminated(game::MuGameEnvWrapper) = false 
GI.game_terminated(game::MuGameEnvWrapper) = GI.game_terminated(GI.init(GI.spec(game.game),game.curstate))

GI.current_state(game::MuGameEnvWrapper) = game.curstate

GI.available_actions(game::MuGameEnvWrapper) = game.curstate == game.rootstate ? GI.available_actions(game.game) : GI.actions(GI.spec(game.game))

GI.white_playing(game::MuGameEnvWrapper) = game.white_playing

function GI.play!(game::MuGameEnvWrapper, action)
    if haskey(game.lookuptable, (game.curstate, action))
        (game.lastreward, game.curstate) = game.lookuptable[(game.curstate,action)]
    else
        (R, S) = game.dynamics_oracle(game.curstate, action)
        game.lookuptable[(game.curstate, action)] = (R, S)
        (game.lastreward, game.curstate) = (R, S)
        @debug "New entry in lookuptable" game.lookuptable
    end
    game.white_playing = !game.white_playing
end

GI.white_reward(game::MuGameEnvWrapper) = game.lastreward #vulnerable when call white_reward() before play!()

"""
    WARNING
    This function do not clone wrapper, 
    It reset curstate to rootstate and return the same wrapper

    That way there is one lookuptable shared between MCTS simulations
    (copying large lookuptable would be costly)
"""
function GI.clone(game::MuGameEnvWrapper) # make lookuptable shallowcopy, or pass it other way
    game.curstate = game.rootstate
    return game
end


