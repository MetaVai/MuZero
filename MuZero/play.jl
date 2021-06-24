struct MuPlayer{M,R,D} <: AbstractPlayer 
    mcts :: M
    representation_oracle :: R
    dynamics_oracle :: D
    niters :: Int
    timeout :: Union{Float64, Nothing}
    τ :: AbstractSchedule{Float64} # Temperature
    function MuPlayer(mcts::MCTS.Env, representation_oracle, dynamics_oracle; τ, niters, timeout=nothing)
        @assert niters > 0
        @assert isnothing(timeout) || timeout > 0
        new{typeof(mcts), typeof(representation_oracle), typeof(dynamics_oracle)}(mcts, representation_oracle, dynamics_oracle, niters, timeout, τ)
    end
end

function MuPlayer(
    game_spec::AbstractGameSpec, p_oracle, r_oracle, d_oracle, params::MctsParams; timeout=nothing, S=GI.state_type(game_spec)) # TODO automateS
  mcts = MCTS.Env(game_spec, p_oracle,
    gamma=params.gamma,
    cpuct=params.cpuct,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α,
    prior_temperature=params.prior_temperature,
    S=S)
  return MuPlayer(mcts, r_oracle, d_oracle;
    niters=params.num_iters_per_turn,
    τ=params.temperature,
    timeout=timeout)
end

function player_temperature(p::Union{MctsPlayer,MuPlayer}, game, turn)
    return p.τ[turn]
  end
  
function reset_player!(player::Union{MctsPlayer,MuPlayer})
    MCTS.reset!(player.mcts)
end

function normalize_p(P, actions_mask)
    P .*= actions_mask
    sp = sum(P, dims=1)
    P ./= sp .+ eps(eltype(P))
    return P[actions_mask]
end

function AlphaZero.think(p::MuPlayer, game)
    hidden_state = p.representation_oracle(GI.current_state(game))

    (P₀, V₀) = p.mcts.oracle(hidden_state)
    P₀ = normalize_p(P₀, GI.actions_mask(game))
    p.mcts.tree = Dict(hidden_state => MCTS.init_state_info(P₀,V₀,p.mcts.prior_temperature))

    mugame = MuGameEnvWrapper(game,
        CachedOracle(p.dynamics_oracle,Dict{Tuple{typeof(hidden_state), Int},Tuple{Float64, typeof(hidden_state)}}()),
        p.representation_oracle,
        hidden_state, 
        true,
        GI.white_playing(game),
        0.)

    if isnothing(p.timeout) # Fixed number of MCTS simulations
        MCTS.explore!(p.mcts, mugame, p.niters)
    else # Run simulations until timeout
        start = time()
        while time() - start < p.timeout
        MCTS.explore!(p.mcts, mugame, p.niters)
        end
    end
    return MCTS.policy(p.mcts, mugame)
end