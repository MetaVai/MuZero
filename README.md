This package started as Google Summer of Code 2021 project
https://summerofcode.withgoogle.com/projects/#4538531164192768

It is based on [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl),
and is inspired by [muzero-general](https://github.com/werner-duvaud/muzero-general)

To train MuZero on tic tac toe, clone this repo, change branch to MuZero,
```
git clone https://github.com/michelangelo21/MuZero.git
cd MuZero
git checkout MuZero
```
and run
```
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project ./MuZero/scripts/train_tictactoe.jl 
```
then, to observe results run in different terminal:
```
tensorboard --logdir results
```