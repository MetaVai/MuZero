#* Wrapper of GameInterface
# redirects AlphaZero.MCTS calls to MuNetwork

"""
    MuGameEnvWrapper

Wraper aroud GameEnv that redirects GI calls
- `game::GI.AbstractGameEnv`  GameEnv that is wrapped
- `dynamics_oracle`           g(sᵏ⁻¹,aᵏ) -> (rᵏ,sᵏ) that returns reward and next state
- `curstate`                  current (hidden) state of game
- `isrootstate::Bool`         whether or not curstate is the root state of MCTS tree
- `white_playing::Bool`       whether or not white is currently playing
- `lastwr::Float64`           last white reward obtained from dynamics_oracle 
"""
mutable struct MuGameEnvWrapper{State, D} <: GI.AbstractGameEnv
  game :: GI.AbstractGameEnv
  dynamics_oracle :: D # g(s,a) -> r,s 
  curstate :: State
  isrootstate :: Bool
  white_playing :: Bool # MuWrapper changes player each round
  lastwr :: Float64
end

GI.game_terminated(game::MuGameEnvWrapper) = false 
# GI.game_terminated(game::MuGameEnvWrapper) = GI.game_terminated(GI.init(GI.spec(game.game),game.curstate))

GI.current_state(game::MuGameEnvWrapper) = game.curstate

GI.available_actions(game::MuGameEnvWrapper) = game.isrootstate ? GI.available_actions(game.game) : GI.actions(GI.spec(game.game))

GI.white_playing(game::MuGameEnvWrapper) = game.white_playing

function GI.play!(game::MuGameEnvWrapper, action)
  # dynamics returns reward from the perspective of last player (rᵂ , sᴮ) <- g(sᵂ, aᵂ)
  (game.lastwr, game.curstate) = game.dynamics_oracle(game.curstate, action)
  if GI.two_players(GI.spec(game.game))
    game.white_playing || (game.lastwr = -game.lastwr) # then it is changed if black was playing and saved as lastwr
    game.white_playing = !game.white_playing # player change
  end
  game.isrootstate = false
end

GI.white_reward(game::MuGameEnvWrapper) = game.lastwr #vulnerable when call white_reward() before play!()

function GI.clone(game::MuGameEnvWrapper)
  MuGameEnvWrapper(
    GI.clone(game.game),
    game.dynamics_oracle,
    game.curstate,
    game.isrootstate,
    game.white_playing,
    game.lastwr)
end


