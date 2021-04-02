# MuZero
- [ ] main muzero() function
- [ ] SharedStorage 
- [ ] ReplayBuffer - memory.jl, trace.jl
- [ ] initial inference (0 --h-> s₀ --f-> p,v)
- [ ] recurrent inference (sᵢ,a --g-> r,sᵢ₊₁ --f-> p,v)
- [ ] self-play (data generation)
    - [ ] play_game() - mostly implemented in play.jl
    - [ ] MCTS - mcts.jl has different interface than pseudocode.py
        - [ ] exploration noise - Dirichlet, (think it's done - mcts.jl)
        - [ ] ucb - done in uct_scores() function (mcts.jl)
        - [ ] action selection - softmax with temperature
        - [ ] backpropagate, normalize Q to [0,1]
- [ ] training (weights optimization) - training.jl has some (most?) features implemented, need to expand for 3 networks
    - [ ] train!()
    - [ ] batch sampler 
    - [ ] make_target - TD
    - [ ] update weights
    - [ ] loss function - sum(l_reward + l_value + l_policy + c||θ||²)
- [ ] networks (use AbstractNetwork interface to implement default ones)
    - [ ] f-network (s -> p,v) - similar to AlphaZero one
    - [ ] dynamics function g-network (s,a -> r,s) - two-headed nns
    - [ ] representation function h-network (0 -> s) - conv default (for Atari games details in paper)



- [ ] use Atari Envs
    - [ ] https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl
    ALE(cpp) -> ArcadeLearningEnvironment.jl -> ReinforcementLearningEnvironments.jl -> CommonRLInterface.jl
    (common_rl_intrf -> GI is already implemented)


### Mostly based on:
- DeepMind's nature paper - Schrittwieser et al. (2020) 
https://www.nature.com/articles/s41586-020-03051-4
- David Foster's "MuZero:The Walkthrough"
https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a
- pseudocode provided with paper 
https://arxiv.org/src/1911.08265v1/anc/pseudocode.py

