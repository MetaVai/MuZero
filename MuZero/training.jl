using DataStructures:CircularBuffer

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
  curnns::MuNetwork
  bestnns::MuNetwork
  memory::CircularBuffer{MuTrace{State}}
  itc::Int
  function MuEnv(
    gspec::AbstractGameSpec,
    params, curnns, bestnns=deepcopy(curnns), experience=[], itc=0)
    msize = max(params.mem_buffer_size, length(experience))
    memory = CircularBuffer{MuTrace{GI.state_type(gspec)}}(msize)
    return new{typeof(gspec),typeof(curnns),GI.state_type(gspec)}(
        gspec, params, curnns, bestnns, memory, itc)
  end
end


# simulate `num_workers` games on one machine using `Threads.nthreads()` Threads
function simulate(simulator::Simulator, gspec::AbstractGameSpec, p)
  total_simulated = Threads.Atomic{Int64}(0)
  return @withprogress name="simulating" AlphaZero.Util.mapreduce(1:p.num_games, p.num_workers, vcat, []) do
  # for _ in 1:p.num_games
    oracles = simulator.make_oracles()
    player = simulator.make_player(oracles)
    function simulate_game(sim_id)
      # worker_sim_id += 1
      # Switch players' colors if necessary: "_pf" stands for "possibly flipped"
      # if isa(player, TwoPlayers) && p.alternate_colors
      #   colors_flipped = sim_id % 2 == 1
      #   player_pf = colors_flipped ? flipped_colors(player) : player
      # else
      #   colors_flipped = false
      #   player_pf = player
      # end
      player_pf = player
      # Play the game and generate a report
      trace = AlphaZero.play_game(gspec, player_pf, flip_probability=p.flip_probability)
      # report = simulator.measure(trace, colors_flipped, player)
      # Reset the player periodically
      # if !isnothing(p.reset_every) && worker_sim_id % p.reset_every == 0
      #   reset_player!(player)
      # end
      # Signal that a game has been simulated
      # game_simulated() 
      Threads.atomic_add!(total_simulated, 1) # don't know if there will be tension between adding, and getting number, but compiler should speedup it, so prevents it
      @logprogress total_simulated[] / p.num_games #progressbar #! uncomment
      # @info "selfplay progress" total_simulated[] / p.num_games
      return trace
    end
    return (process = simulate_game, terminate = (() -> nothing))
    # simulate_game(1)
  end
end

#####
##### Evaluating networks
#####

# Have a "contender" network play against a "baseline" network (params::ArenaParams)
# Return (rewards vector, redundancy)
# Version for two-player games 
#TODO incorporate nns pit
# function pit_networks(gspec, contender, baseline, params, handler)
#   make_oracles() = (
#     deepcopy(contender),
#     deepcopy(baseline))
#   simulator = Simulator(make_oracles, record_trace) do oracles
#     white = MuPlayer(gspec, oracles[1], params.mcts)
#     black = MuPlayer(gspec, oracles[2], params.mcts)
#     return TwoPlayers(white, black)
#   end
#   samples = simulate(
#     simulator, gspec, params.sim)
#   return rewards_and_redundancy(samples, gamma=params.mcts.gamma) #TODO create analyzer function
# end


##### Main training loop

function self_play_step!(env::MuEnv)
  params = env.params.self_play
  make_oracle() = deepcopy(env.bestnns)
  simulator = Simulator(make_oracle, nothing) do nns
    return MuPlayer(nns, params.mcts)
  end
  traces = simulate(simulator, gspec, params.sim)
  # @info "Self Play Step" lost, draw, won=(count_wins(traces) ./ length(traces))

  append!(env.memory, traces)
  @info "Self Play Step" stage="finished" lost, draw, won=(count_wins(traces)./length(traces))

end

function learning_step!(env::MuEnv)
  lp = env.params.learning_params
  ap = nothing

  #? symmetries 

  #TODO Trainer as separate actor
  tr = MuTrainer(env.gspec, env.curnns, env.memory, env.params.learning_params, Flux.ADAM(lp.learning_rate))
  nbatches = lp.batches_per_checkpoint

  @progress "learning step (epoch)" for _ in 1:lp.num_checkpoints
    update_weights!(tr, nbatches)
    if isnothing(ap)
      # env.curnns = tr.nns # trainer operate on shallow copy of curnns
      env.bestnns = deepcopy(env.curnns)
      # @info "nns replaced" nns_replaced=true
    else
    end
  end
  @info "Learning Step" stage="finished"
end

# one machine - learning
# others - self_play
#? tasks while-loop for async (and distributed) (pure Julia, Actors.jl, Jun's Oolong.jl, Dagger.jl)
function train!(env::MuEnv; benchmark=[])
  while env.itc < env.params.num_iters
    @info "Training" stage="starting iteration" env.itc
    _, time_self_play = @timed self_play_step!(env) #TODO create custom time loggers
    _, time_learning = @timed learning_step!(env)
    @info "Training" stage="iteration finished" env.itc time_self_play time_learning
    run_duel(env, benchmark) # compare MuZero with pure MCTS or other algorithm 
    env.itc += 1
  end
  @info "Training" stage="Finished"
end
