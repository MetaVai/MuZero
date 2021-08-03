# TODO move this constructor to mcts.jl 
function MCTS.Env(gspec::AbstractGameSpec, oracle, params::MctsParams; S=nothing)
	return MCTS.Env(gspec, oracle;
	gamma=params.gamma,
	cpuct=params.cpuct,
	noise_ϵ=params.dirichlet_noise_ϵ,
	noise_α=params.dirichlet_noise_α,
	prior_temperature=params.prior_temperature,
	S=S)
end

"""
		MuPlayer{P,D,R} <: AbstractPlayer

- `prediction_oracle` 		f(sᵏ) -> (pᵏ,vᵏ)
- `dynamics_oracle`				g(sᵏ⁻¹,aᵏ) -> (rᵏ,sᵏ)
- `representation_oracle` h(o) -> (s⁰)
- `mcts_params::MctsParams`
- `timeout::Union{Float64, Nothing}` time that MCTS has for thinking
		nothing means infinite (till num_iters_per_turn ends)
"""
struct MuPlayer{P,D,R} <: AbstractPlayer 
	# mcts :: M
	prediction_oracle :: P
	dynamics_oracle :: D
	representation_oracle :: R
	mcts_params :: MctsParams
	timeout :: Union{Float64, Nothing}
	# niters :: Int
	# τ :: AbstractSchedule{Float64} # Temperature
	# function MuPlayer(mcts::MCTS.Env, representation_oracle, dynamics_oracle; τ, niters, timeout=nothing)
	function MuPlayer(f, g, h, mcts_params::MctsParams; timeout=nothing)
		@assert mcts_params.num_iters_per_turn > 0
		@assert isnothing(timeout) || timeout > 0
		new{typeof(f), typeof(g), typeof(h)}(f, g, h, mcts_params, timeout)
	end
end

# constructor with MuNetwork
function MuPlayer(
	μnetwork::Union{MuNetwork,NamedTuple}, params::MctsParams; timeout=nothing)
  return MuPlayer(μnetwork.f, μnetwork.g, μnetwork.h, params;
	# niters=params.num_iters_per_turn,
	# τ=params.temperature,
	timeout=timeout)
end

function AlphaZero.player_temperature(p::MuPlayer, game, turn)
	return p.mcts_params.temperature[turn]
end
  
# function reset_player!(player::Union{MctsPlayer,MuPlayer})
# 	MCTS.reset!(player.mcts)
# end

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
# compute_rootvalue(ri::MCTS.StateInfo, γ) = max(st.W/st.N for st in ri.stats)
compute_rootvalue(ri::MCTS.StateInfo) = sum(st.W for st in ri.stats) / MCTS.Ntot(ri)
# TODO compare max and sum

"""
		think(::MuPlayer, game)

Return avaliable actions, policy, and rootvalue
"""
function AlphaZero.think(p::MuPlayer, game)
	#initial inference:  h(o) → s⁰,  f(s⁰) → (p⁰,v⁰)
	rootstate = p.representation_oracle(GI.current_state(game))
	(P⁰, V⁰) = p.prediction_oracle(rootstate)
	actions_mask = GI.actions_mask(game)
	P⁰ = normalize_p(P⁰, actions_mask)[actions_mask]

	mcts = MCTS.Env(GI.spec(game), p.prediction_oracle, p.mcts_params, S=typeof(rootstate))
	mcts.tree[rootstate] = MCTS.init_state_info(P⁰,V⁰,mcts.prior_temperature)

	mugame = MuGameEnvWrapper(
		game,
		#	CachedOracle(oracle, Statetype, Actiontype, Rewardtype)
		CachedOracle(p.dynamics_oracle,typeof(rootstate),GI.action_type(mcts.gspec),Float64),
		rootstate, 
		true, #isroot
		GI.white_playing(game),
		0.)
	niters = p.mcts_params.num_iters_per_turn - 1 # -1, because (P⁰,V⁰) is already there
	if isnothing(p.timeout) # Fixed number of MCTS simulations
		MCTS.explore!(mcts, mugame, niters) 
	else # Run simulations until timeout
		start = time()
		while time() - start < p.timeout
		MCTS.explore!(mcts, mugame, niters)
		end
	end
	rootvalue = compute_rootvalue(mcts.tree[rootstate])
	actions, π_target = MCTS.policy(mcts, mugame)
	return actions, π_target, rootvalue
end

"""
		play_game(gspec, ::MuPlayer)

overloaded AlphaZero.play_game() function, 
that additionally push selected action, and rootvalue into trace
"""
function AlphaZero.play_game(gspec, player::Union{MuPlayer,MinMax.Player,MctsPlayer}; flip_probability=0., initstate=nothing)
  game = isnothing(initstate) ? GI.init(gspec) : GI.init(gspec, initstate)
	# @debug "initstate" initstate GI.current_state(game)
  trace = MuTrace(GI.current_state(game))
  while true
    if GI.game_terminated(game)
      return trace
    end
    if !iszero(flip_probability) && rand() < flip_probability
      GI.apply_random_symmetry!(game)
    end
    actions, π_target, rootvalue = think(player, game)
		rootvalue = GI.white_playing(game) ? rootvalue : -rootvalue# ? reversing rootvalue wp?
    τ = AlphaZero.player_temperature(player, game, length(trace))
    π_sample = apply_temperature(π_target, τ)
    a = actions[Util.rand_categorical(π_sample)]
    GI.play!(game, a)
    push!(trace, π_target, GI.white_reward(game), GI.current_state(game), a, rootvalue) # pᵏ, rᵏ, sᵏ, aᵏ, vᵏ
  end
end