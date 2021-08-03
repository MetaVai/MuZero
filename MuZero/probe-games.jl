module ProbeGames

include("../games/probe-games/a1r1/main.jl")
export A1R1

include("../games/probe-games/a1o2r2/main.jl")
export A1O2R2

include("../games/probe-games/a1r1s2/main.jl")
export A1R1S2

include("../games/probe-games/a2r2/main.jl")
export A2R2

include("../games/probe-games/two_player/main.jl")
export TWOPLAYER

include("../games/probe-games/simpler_tictactoe/main.jl")
export SimplerTicTacToe


games = Dict(
  "a1r1" => A1R1.GameSpec(),
  "a1o2r2" => A1O2R2.GameSpec(),
  "a1r1s2" => A1R1S2.GameSpec(),
  "a2r2" => A2R2.GameSpec(),
  "twoplayer" => TWOPLAYER.GameSpec(),
  "simplertictactoe" => SimplerTicTacToe.GameSpec()
)
end #ProbeGames