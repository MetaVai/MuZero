# simple two player game, when someone choose action 2, game terminates, and that player wins
import AlphaZero.GI
# using Flux: onehot

const WHITE = true
const BLACK = false

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv 
  terminated :: Bool
  curplayer :: Bool
end

const INITSTATE = (; terminated=false, curplayer=BLACK)

GI.init(::GameSpec, state=INITSTATE) = GameEnv(state.terminated, state.curplayer)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = true

const NUM_ACTIONS = 9
const ACTIONS = collect(1:NUM_ACTIONS)

GI.actions(::GameSpec) = ACTIONS

GI.actions_mask(g::GameEnv) = trues(NUM_ACTIONS)

GI.current_state(g::GameEnv) = (; terminated=g.terminated, curplayer=g.curplayer)

GI.white_playing(g::GameEnv) = g.curplayer

GI.game_terminated(g::GameEnv) = g.terminated

function GI.white_reward(g::GameEnv)
  if g.terminated
    return g.curplayer ? -1. : 1.
  else
    return 0.
  end
end

function GI.play!(g::GameEnv, action)
  if action == 6
    g.terminated = true
  end
  g.curplayer = !g.curplayer
end

GI.vectorize_state(::GameSpec, state) = Float32[state.terminated]