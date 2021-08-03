#Two actions, zero observation, one timestep long, action-dependent +1/-1 reward: The first 
#env to exercise the policy! If my agent can't learn to pick the better action, there's 
#something wrong with either my advantage calculations, my policy loss or my policy update. 
#That's three things, but it's easy to work out by hand the expected values for each one and
# check that the values produced by your actual code line up with them.
import AlphaZero.GI
using Flux: onehot
struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv 
  terminated :: Bool
  numero :: Int
end

const INITSTATE = (; terminated=false, numero=0)

GI.init(::GameSpec, state=INITSTATE) = GameEnv(state.terminated, state.numero)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = false


const ACTIONS = collect(1:2)

GI.actions(::GameSpec) = ACTIONS

GI.actions_mask(g::GameEnv) = trues(2)

GI.current_state(g::GameEnv) = (; terminated=g.terminated, numero=g.numero)

GI.white_playing(g::GameEnv) = true

GI.game_terminated(g::GameEnv) = g.terminated

function GI.white_reward(g::GameEnv)
  if g.terminated
    return g.numero == 1 ? 1. : -1.
  else
    return 0.
  end
end

function GI.play!(g::GameEnv, action)
  g.numero += action
  g.terminated = true
end

GI.vectorize_state(::GameSpec, state) = collect(onehot(state.numero, 0:2))