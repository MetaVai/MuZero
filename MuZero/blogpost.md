# GSoC midway update
A half of summer has just flown by, so it's time to prepare some report describing current state of my GSoC project. Below are highlighted more interesting things about MuZero implementation, and the full code is available in [this repo](https://github.com/michelangelo21/AlphaZero.jl/tree/MuZero). 

## MCTS
Both AlphaZero and MuZero use MonteCarloTreeSearch, so it was convenient to reuse this part. The difference is that while AlphaZero operates on copy of GameEnvironment (observations as states, known available actions, transitions etc), MuZero's MCTS works in hiddenspace (hiddenstate is generated from observation via RepresentationOracle) and calls DynamicsOracle for state transitions and rewards.

MuZero uses AlphaZero.jl's MCTS through the GameEnv wrapper - redirects MCTS's Game Interface
calls to Dynamics Oracle (neural network).

Transition history $(s^{k-1}, a^{k})\Rightarrow(r^k, s^k)$ is stored in a lookuptable in CachedOracle - only calls nn if state-action-pair isn't in a lookup table. That abstraction enables clarity of code. 

Action mask is applied only on rootstate, so the distinction between root and non-root state is needed.

AlphaDynamics, and AlphaRepresentation are provided to mimics behaviour of AlphaZro. 

### Changes to AlphaZero.jl's MCTS
- added depth counter, and maximum depth to prevent infinite recursion when using AlphaDynamics (action mask isn't applied on states other than root, and choosing illegal action returns the same state, which is already in MCTS tree); this is ad hoc solution, see [related discussion](https://github.com/jonathan-laurent/AlphaZero.jl/issues/47 "StackOverflowError on cyclic state graph game #47")
- moved StateType to MCTS.Env's constructor parameter (from its body)

## Self-play
Most notable difference here is dispatched `think!` function. 

Firstly it obtains hiddenstate from RepresentationOracle $h(o)\rightarrow s^0$, then gets value and normalized policy from PredictionOracle $f(s^0) \rightarrow (p^0, v^0)$ - this is called *initial reference* in DeepMind's pseudocode.

Next, new mcts is created with rootstate info. That way action mask is already applied and don't need to distinguish between root and other states when calling `init_state_info()`.
Then `game` is wrapped in `MuGameEnvWrapper` with cached DynamicsOracle, and MCTS is started. *recurrent inference* is split between MCTS ($f$) and wrapper ($g$)


During self-play phase MuZero plays against itself (using MCTS) and collecting whole trace. `play!` and function was dispatched to collect selected action and rootvalue (together with state, policy and reward already collected by AlphaZero's `play!`)

Games are simulated using multiple threads like in AlphaZero.jl

## Learning
Target samples are generated like in pseudocode. Then they are batched for better future usage in GPU.

Losses are computed like in DeepMind's paper appendix for  ($l_p$=crossentropy, $l_v$=mse, $l_r$=0, with $L2$ params regularization)

## Misc
- statistics are logged to TensorBoard (using [TensorBoard.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl))
- progress is showed during self-play, and learning phase along with time left (using [ProgressLogging.jl](https://github.com/JuliaLogging/ProgressLogging.jl))
- self-play step and learning step are currently performed one after another, but in the future they should run concurrently
- for now two-headed networks with dense layers are implemented, with ResNet support planned in upcoming weeks

## What next?
- debugging - make sure that code runs well without runtime bugs
- GPU support - more advanced models would run till the end of time without GPU
- clear documentation - what's the point of package existence if nobody can use it?

Phase one of GSoC was a great experience for me. I'm looking forward for the the second half, as it will be even more challenging, thus more exciting.