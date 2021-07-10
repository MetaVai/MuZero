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

# normalize policy - probs of legal actions sum up to 1, and illegals are 0
function normalize_p(P, actions_mask)
	P = P .* actions_mask # Zygote doesn't work with .*= 
	sp = sum(P, dims=1)
	P = P ./ (sp .+ eps(eltype(P)))
	return P
end

# ? does rootinfo.vest should be there, or just
# ? sum without γ
# compute_rootvalue(rootinfo::MCTS.StateInfo, γ) = rootinfo.Vest + γ*sum([st.W for st in rootinfo.stats])
compute_rootvalue(ri::MCTS.StateInfo, γ) = max([st.W/st.N for st in ri.stats])
compute_rootvalue(ri::MCTS.StateInfo, γ) = sum([st.W for st in ri.stats]) / MCTS.Ntot(ri)
#TODO compare max and sum

function AlphaZero.think(p::MuPlayer, game)
	rootstate = p.representation_oracle(GI.current_state(game))

	(P₀, V₀) = p.mcts.oracle(rootstate)
	actions_mask = GI.actions_mask(game)
	P₀ = normalize_p(P₀, actions_mask)[actions_mask]
	# TODO create new mcts, reset mcts, make more clear MCTS.reset!()
	MCTS.reset!(p.mcts)
	p.mcts.tree[rootstate] = MCTS.init_state_info(P₀,V₀,p.mcts.prior_temperature)

	mugame = MuGameEnvWrapper(
		game,
		#	CachedOracle(oracle, Statetype, Actiontype, Rewardtype)
		CachedOracle(p.dynamics_oracle,typeof(rootstate),GI.action_type(p.mcts.gspec),Float64),
		rootstate, 
		true,
		GI.white_playing(game),
		0.)

	if isnothing(p.timeout) # Fixed number of MCTS simulations
		MCTS.explore!(p.mcts, mugame, p.niters-1) # -1, because (P₀,V₀) is already there
	else # Run simulations until timeout
		start = time()
		while time() - start < p.timeout
		MCTS.explore!(p.mcts, mugame, p.niters-1)
		end
	end
	rootvalue = compute_rootvalue(p.mcts.tree[rootstate], p.mcts.gamma)
	# @info rootvalue	
	# @info p.mcts.tree[rootstate]
	# rootvalue = 0.0
	actions, π_target = MCTS.policy(p.mcts, mugame)
	return actions, π_target, rootvalue
end

function play_game(gspec, player::MuPlayer; flip_probability=0.)
  game = GI.init(gspec)
  trace = MuTrace(GI.current_state(game))
  while true
    if GI.game_terminated(game)
      return trace
    end
    if !iszero(flip_probability) && rand() < flip_probability
      GI.apply_random_symmetry!(game)
    end
    actions, π_target, rootvalue = think(player, game)
    τ = player_temperature(player, game, length(trace))
    π_sample = apply_temperature(π_target, τ)
    a = actions[Util.rand_categorical(π_sample)]
    GI.play!(game, a)
    push!(trace, π_target, GI.white_reward(game), GI.current_state(game), a, rootvalue)
  end
end