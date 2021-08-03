#One action, zero-then-one observation, two timesteps long, +1 reward at the end: If my agent 
#can learn the value in (2.) but not this one, it must be that my reward discounting is broken.
import AlphaZero.GI
using Flux: onehot
struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv 
  terminated :: Bool
  numero :: Int
end

const INITSTATE = (; terminated=false, numero=1)

GI.init(::GameSpec, state=INITSTATE) = GameEnv(state.terminated, state.numero)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = false


const ACTIONS = collect(1:1)

GI.actions(::GameSpec) = ACTIONS

GI.actions_mask(g::GameEnv) = [true]

GI.current_state(g::GameEnv) = (; terminated=g.terminated, numero=g.numero)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = g.terminated

GI.white_reward(g::GameEnv) = g.terminated ? 1. : 0.

function GI.play!(g::GameEnv, action)
  if g.numero >= 2
    g.terminated = true
  end
  g.numero += 1
end

GI.vectorize_state(::GameSpec, state) = collect(onehot(state.numero, 1:2))