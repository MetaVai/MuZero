# One action, zero observation, one timestep long, +1 reward every timestep: This isolates 
# the value network. If my agent can't learn that the value of the only observation it ever 
# sees it 1, there's a problem with the value loss calculation or the optimizer.
import AlphaZero.GI

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv 
  terminated :: Bool
end

const INITSTATE = (; terminated=false)

GI.init(::GameSpec, state=INITSTATE) = GameEnv(state.terminated)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = false


const ACTIONS = collect(1:1)

GI.actions(::GameSpec) = ACTIONS

GI.actions_mask(g::GameEnv) = [true]

GI.current_state(g::GameEnv) = (; terminated=g.terminated)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = g.terminated

GI.white_reward(g::GameEnv) = 1.

function GI.play!(g::GameEnv, action)
  g.terminated = true
end

GI.vectorize_state(::GameSpec, state) = Float32[!state.terminated]