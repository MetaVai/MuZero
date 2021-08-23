using DataStructures:CircularBuffer
using Statistics: stdm
import FileIO, JLD2
using Dates: now, format
import Folds

struct MuParams
  self_play
  learning_params
  arena
  num_iters::Int
  mem_buffer_size
end

# mutable struct MuEnv{GameSpec, Network, State}
#     gspec :: GameSpec
#     params :: MuParams
#     curnns :: Network
#     bestnns :: Network
#     memory :: MuMemoryBuffer{GameSpec, State}
#     itc :: Int
#     function MuEnv(
#         gspec::AbstractGameSpec,
#         params, curnns, bestnns=deepcopy(curnns), experience=[], itc=0)
#         msize = max(params.mem_buffer_size, length(experience))
#         memory = MuMemoryBuffer(gspec, msize, experience)
#         return new{typeof(gspec), typeof(curnns), GI.state_type(gspec)}(
#             gspec, params, curnns, bestnns, memory, itc)
#     end
# end

mutable struct MuEnv{GameSpec,Network,State}
  gspec::GameSpec
  params::MuParams
  curnns::Network
  bestnns::Network
  memory::CircularBuffer{MuTrace{State}}
  itc::Int
  function MuEnv(
    gspec::AbstractGameSpec,
    params, curnns; bestnns=deepcopy(curnns), experience=[], itc=0)
    msize = max(params.mem_buffer_size, length(experience))
    memory = CircularBuffer{MuTrace{GI.state_type(gspec)}}(msize)
    append!(memory, experience)
    return new{typeof(gspec),typeof(curnns),GI.state_type(gspec)}(
        gspec, params, curnns, bestnns, memory, itc)
  end
end

# #! ORIGINAL 
# # simulate `num_workers` games on one machine using `Threads.nthreads()` Threads
# function simulate(simulator::Simulator, gspec::AbstractGameSpec, p)
#   total_simulated = Threads.Atomic{Int64}(0) # used for progress logging
#   # return @withprogress name="simulating" AlphaZero.Util.mapreduce(1:p.num_games, p.num_workers, vcat, []) do
#   # @withprogress for i in 1:p.num_games # easier for debugging
#   # @withprogress name="simulating" map(1:p.num_games) do i # easier for debugging
#     oracles = simulator.make_oracles()
#     # player = simulator.make_player(oracles)
#     spawn_oracles, done = batchify_oracles(oracles; p.num_workers, p.batch_size, p.fill_batches)
#     function simulate_game(sim_id, player)
#       # worker_sim_id += 1
#       # Switch players' colors if necessary: "_pf" stands for "possibly flipped"
#       if isa(player, TwoPlayers) && p.alternate_colors
#         colors_flipped = sim_id % 2 == 1
#         player_pf = colors_flipped ? AlphaZero.flipped_colors(player) : player
#       else
#         colors_flipped = false
#         player_pf = player
#       end
#       # Play the game and generate a report
#       trace = AlphaZero.play_game(gspec, player_pf, flip_probability=p.flip_probability)
#       # report = simulator.measure(trace, colors_flipped, player)
#       # Reset the player periodically
#       # if !isnothing(p.reset_every) && worker_sim_id % p.reset_every == 0
#         reset_player!(player)
#       # end
#       # Signal that a game has been simulated
#       # game_simulated() 

#       # @info "selfplay progress" total_simulated[] / p.num_games
#       return (; trace, colors_flipped)
#     end
#   # TensorBoardLogger doesn't work with logging on multiple threads
#   #// return @withprogress Folds.map(1:p.num_games) do i # easier for debugging
#   return Folds.mapreduce(vcat, 1:p.num_workers, init=[]) do worker_id # easier for debugging
#     # return (process = simulate_game, terminate = (() -> nothing))
#     oracles = spawn_oracles()
#     player = simulator.make_player(oracles)
#     local_result = []
#     while total_simulated[] + worker_id <= p.num_games # could be easly changed to simulate[]===true, for async
#       Threads.atomic_add!(total_simulated, 1) # don't know if there will be tension between adding, and getting number, but compiler should speedup it, so prevents it
#       push!(local_result, simulate_game(length(local_result), player))
#       # @debug Threads.threadid() worker_id length(local_result) 
#       #// @logprogress total_simulated[] / p.num_games #progressbar #! uncomment
#     end
#     done()
#     return local_result
#   end
# end


"""
    simulate(::Simulator, ::AbstractGameSpec; ::SimParams; <kwargs>)

Play a series of games using a given [`Simulator`](@ref).

# Keyword Arguments

  - `game_simulated` is called every time a game simulation is completed
    (with no arguments)

# Return

Return a vector of objects computed by `simulator.measure`.
"""
function simulate(
    simulator::Simulator,
    gspec::AbstractGameSpec,
    p::SimParams)

  oracles = simulator.make_oracles()
  spawn_oracles, done = batchify_oracles(oracles; p.num_workers, p.batch_size, p.fill_batches)
  # spawn_oracles, done = simulator.make_oracles, ()->nothing
  return Util.mapreduce(1:p.num_games, p.num_workers, vcat, []) do
    oracles = spawn_oracles()
    player = simulator.make_player(oracles)
    worker_sim_id = 0
    # For each worker
    function simulate_game(sim_id)
      worker_sim_id += 1
      # Switch players' colors if necessary: "_pf" stands for "possibly flipped"
      if isa(player, TwoPlayers) && p.alternate_colors
        colors_flipped = sim_id % 2 == 1
        player_pf = colors_flipped ? AlphaZero.flipped_colors(player) : player
      else
        colors_flipped = false
        player_pf = player
      end
      # Play the game and generate a report
      trace = play_game(gspec, player_pf, flip_probability=p.flip_probability)
      # Reset the player periodically
      if !isnothing(p.reset_every) && worker_sim_id % p.reset_every == 0
        reset_player!(player)
      end
      # Signal that a game has been simulated
      return (; trace, colors_flipped)
    end
    return (process=simulate_game, terminate=done)
  end
end

# #! FOR MINMAX RANDOM STARTING STATES
# # simulate `num_workers` games on one machine using `Threads.nthreads()` Threads
# function simulate(simulator::Simulator, gspec::AbstractGameSpec, p)
#   total_simulated = Threads.Atomic{Int64}(0)
#   return @withprogress name="simulating" AlphaZero.Util.mapreduce(1:p.num_games, p.num_workers, vcat, []) do
#   # for _ in 1:p.num_games # easier for debugging
#     oracles = simulator.make_oracles()
#     player = simulator.make_player(oracles)
#     function simulate_game(sim_id)
#       # worker_sim_id += 1
#       # Switch players' colors if necessary: "_pf" stands for "possibly flipped"
#       if isa(player, TwoPlayers) && p.alternate_colors
#         colors_flipped = sim_id % 2 == 1
#         player_pf = colors_flipped ? AlphaZero.flipped_colors(player) : player
#       else
#         colors_flipped = false
#         player_pf = player
#       end
#       # Play the game and generate a report
#       rinitstate = rand(AlphaZero.play_game(gspec, RandomPlayer()).states[1:end-1])
#       # @info rinitstate
#       trace = AlphaZero.play_game(gspec, player_pf, flip_probability=p.flip_probability, initstate=rinitstate)
#       # report = simulator.measure(trace, colors_flipped, player)
#       # Reset the player periodically
#       # if !isnothing(p.reset_every) && worker_sim_id % p.reset_every == 0
#       #   reset_player!(player)
#       # end
#       # Signal that a game has been simulated
#       # game_simulated() 
#       Threads.atomic_add!(total_simulated, 1) # don't know if there will be tension between adding, and getting number, but compiler should speedup it, so prevents it
#       @logprogress total_simulated[] / p.num_games #progressbar #! uncomment
#       # @info "selfplay progress" total_simulated[] / p.num_games
#       return (; trace, colors_flipped)
#     end
#     return (process = simulate_game, terminate = (() -> nothing))
#     # simulate_game(1)
#   end
# end


#####
##### Evaluating networks
#####

# Have a "contender" network play against a "baseline" network (params::ArenaParams)
# Return (rewards vector, redundancy)
# Version for two-player games 
#TODO incorporate nns pit
function pit_networks(gspec, contender, baseline, params)
  function make_oracles()
    nnc = deepcopy(contender)|>params.device|>Flux.testmode!
    nnb = deepcopy(baseline)|>params.device|>Flux.testmode!
    return ((InitialOracle(nnc), RecurrentOracle(nnc)),
            (InitialOracle(nnb), RecurrentOracle(nnb)))
  end
  simulator = Simulator(make_oracles, record_trace) do oracles
    white = MuPlayer(oracles[1], params.mcts)
    black = MuPlayer(oracles[2], params.mcts)
    return TwoPlayers(white, black)
  end
  samples = simulate(
    simulator, gspec, params.sim)
  return rewards_and_redundancy(samples, gamma=params.mcts.gamma) #TODO create analyzer function
end


##### Main training loop

function self_play_step!(env::MuEnv)
  @info "Self Play Step" stage="started"
  params = env.params.self_play
  # make_oracle() = deepcopy(rand([env.bestnns, env.curnns]))|>params.device|>Flux.testmode!
  function make_oracle()
    nns = deepcopy(rand([env.bestnns, env.curnns]))|>params.device|>Flux.testmode!
    return (InitialOracle(nns), RecurrentOracle(nns))
  end
  simulator = Simulator(make_oracle, record_trace) do nns
    return MuPlayer(nns, params.mcts) 
    # return MinMax.Player(depth=7, amplify_rewards=false, τ=0.25)
    # return MctsPlayer(gspec, MCTS.RolloutOracle(gspec), params.mcts)
  end
  results = simulate(simulator, env.gspec, params.sim)

  for r in results
    push!(env.memory, r.trace)
  end
  # append!(env.memory, (r.trace for r in results)) # compare memory efficiency with for push!
  @info "Self Play Step" stage="finished"
end

function learning_step!(env::MuEnv)
  @info "Learning Step" stage="started"
  lp = env.params.learning_params
  ap = env.params.arena

  #? symmetries 

  #TODO Trainer as separate actor
  tr = MuTrainer(env.gspec, env.curnns|>lp.device|>Flux.trainmode!, env.memory, env.params.learning_params, env.params.learning_params.opt)
  nbatches = lp.batches_per_checkpoint

  # for _ in 1:lp.num_checkpoints
  @progress "learning step (epoch)" for _ in 1:lp.num_checkpoints
    update_weights!(tr, nbatches)
    if isnothing(ap)
      # env.curnns = tr.nns # trainer operate on shallow copy of curnns
      env.bestnns = deepcopy(tr.nns)
      # @info "nns replaced" nns_replaced=true
    else
      r_cnn, redundancy = pit_networks(env.gspec, tr.nns, env.bestnns, ap)
      rewards_curnn_mean = mean(r_cnn)
      if rewards_curnn_mean >= ap.update_threshold
        env.bestnns = deepcopy(tr.nns)
      end
      @info "Learning Step" rewards_curnn_mean curnn_wins=count(isone,r_cnn)/length(r_cnn)
    end
  end
  @info "Learning Step" stage="finished"
end

function memory_analysis(memory)
  unique_games = unique(t.states for t in memory)
  unique_states = unique(s for g in unique_games for s in g)
  num_unique_toplay_white = count(s.curplayer==true for s in unique_states)

  # num_toplay_white = count(s.curplayer for t in memory for s in t.states)

  last_rewards = (last(t.rewards) for t in memory)
  last_rewards_mean = mean(last_rewards)
  percentage_draws = count(iszero, last_rewards) / length(last_rewards) #TODO log
  percentage_white_wins = count(isone, last_rewards) / length(last_rewards)
  percentage_white_loses = count(==(-1), last_rewards) / length(last_rewards)
  # @assert percentage_draws + percentage_white_wins + percentage_white_loses ≈ 1

  mean_trace_length = mean(length(t) for t in memory)
  
  all_rootvalues = (r for t in memory for r in t.rootvalues)

  all_rootvalues_max = maximum(all_rootvalues)
  all_rootvalues_min = minimum(all_rootvalues)

  all_rootvalues_mean = mean(all_rootvalues)
  all_rootvalues_std = stdm(all_rootvalues, all_rootvalues_mean)

  initstate_rootvalues = (first(t.rootvalues) for t in memory)
  initstate_rootvalues_mean = mean(initstate_rootvalues)
  initstate_rootvalues_std = stdm(initstate_rootvalues, initstate_rootvalues_mean)

  entropy(p) = -sum(p .* log.(p .+ eps(eltype(p))))
  all_policies = (p for t in memory for p in t.policies)
  all_policies_entropy_mean = mean(entropy, all_policies)

  initstate_policies = (first(t.policies) for t in memory)
  initstate_policies_entropy_mean = mean(entropy, initstate_policies)

  return (;
    num_traces = length(memory),
    num_unique_games=length(unique_games),
    num_unique_states=length(unique_states),
    num_unique_toplay_white,
    # num_toplay_white,
    # last_rewards_mean,
    percentage_draws,
    percentage_white_wins,
    percentage_white_loses,
    mean_trace_length,
    all_rootvalues_max,
    all_rootvalues_min,
    all_rootvalues_mean,
    all_rootvalues_std,
    initstate_rootvalues_mean,
    initstate_rootvalues_std,
    all_policies_entropy_mean,
    initstate_policies_entropy_mean)
end

# one machine - learning
# others - self_play
#? tasks while-loop for async (and distributed) (pure Julia, Actors.jl, Jun's Oolong.jl, Dagger.jl)
function mutrain!(env::MuEnv; benchmark=(;), path="results/", benchsave_every=50)
  while env.itc <= env.params.num_iters
    @info "Training" stage="starting iteration" env.itc
    _, time_self_play = @timed self_play_step!(env) #TODO create custom time loggers
    mem_report, time_memory_analysis = @timed memory_analysis(env.memory)
    @info "Memory Analysis" memory_analysis(env.memory)...
    _, time_learning = @timed learning_step!(env)
    @info "Training" stage="iteration finished" env.itc time_self_play time_learning time_memory_analysis
    if env.itc % benchsave_every == 0 
      @info "Benchmark" run_duel(env, benchmark)...
      FileIO.save(path*"$(format(now(),"yyyy-mm-ddTHHMM"))_env_$(env.itc).jld2", "env", env)
    end
    env.itc += 1
  end
  @info "Training" stage="Finished"
end
