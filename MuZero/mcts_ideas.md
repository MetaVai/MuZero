Wrapper must be only in mcts, otherwise it should differentiate calls `GI.play!()` between these from MCTS (operating on hidden state) and real moves

add wrapper inside `think(::MctsPlayer,game)`
overload `think()` function, possibly with MuPlayer or something
pass game with hidden state to `MCTS.explore!`
set `root=true` after `clone!`, and `root=false` after `GI.available_actions()` once

change oracle evaluate function