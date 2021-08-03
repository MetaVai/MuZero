# one sided version of tictactoe (  | |  ), when any player has two adjatcent poles it wins
# (so basicly black can't win, and middle cell is win for white)

import AlphaZero.GI
using StaticArrays
# using Flux: onehot

const BOARD_SIDE = 5
const NUM_POSITIONS = BOARD_SIDE

const Player = Bool
const WHITE = true
const BLACK = false

const Cell = Union{Nothing, Player}
const Board = SVector{NUM_POSITIONS, Cell}
const INITIAL_BOARD = Board(repeat([nothing], NUM_POSITIONS))
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=BLACK) # BLACK starts 

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv 
  board :: Board
  curplayer :: Player
end


GI.init(::GameSpec, state=INITIAL_STATE) = GameEnv(state.board, state.curplayer)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = true

function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
end

const ALIGNMENTS = [i-1:i for i in 2:BOARD_SIDE]

function has_won(g::GameEnv, player)
  any(ALIGNMENTS) do al
    all(al) do pos
      g.board[pos] == player
    end
  end
end


const ACTIONS = collect(1:NUM_POSITIONS)

GI.actions(::GameSpec) = ACTIONS

GI.actions_mask(g::GameEnv) = map(isnothing, g.board)

GI.current_state(g::GameEnv) = (board=g.board, curplayer=g.curplayer)

GI.white_playing(g::GameEnv) = g.curplayer

function terminal_white_reward(g::GameEnv)
  has_won(g, WHITE) && return 1.
  has_won(g, BLACK) && return -1.
  isempty(GI.available_actions(g)) && return 0.
  return nothing
end


GI.game_terminated(g::GameEnv) = !isnothing(terminal_white_reward(g))

function GI.white_reward(g::GameEnv)
  z = terminal_white_reward(g)
  return isnothing(z) ? 0. : z
end

function GI.play!(g::GameEnv, pos)
  g.board = setindex(g.board, g.curplayer, pos)
  g.curplayer = !g.curplayer
end

function flip_colors(board)
  flip(cell) = isnothing(cell) ? nothing : !cell
  # Inference fails when using `map`
  return @SVector Cell[flip(board[i]) for i in 1:NUM_POSITIONS]
end

function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  return Float32[
    board[i] == c
    for i in 1:BOARD_SIDE,
        c in [nothing, WHITE, BLACK]]
end

GI.heuristic_value(::GameEnv) = 0.