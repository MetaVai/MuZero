using DataStructures:CircularBuffer

struct MuParams
  self_play_params
  learning_params
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

#TODO add simulator, make_oracles() ... 
function simulate(gspec::AbstractGameSpec, p)
  total_simulated = Threads.Atomic{Int64}(0)
  return @withprogress name="self play step" AlphaZero.Util.mapreduce(1:p.num_games, p.num_workers, vcat, []) do
  # results = map(1:p.num_games) do i
    # make oracles
    representation_oracle = deepcopy(env.bestnns.h)
    # representation_oracle = CachedOracle(representation_oracle, Dict{Tuple{typeof(hidden_state), Int},Tuple{Float64, typeof(hidden_state)}}())
    dynamics_oracle = deepcopy(env.bestnns.g)
    prediction_oracle = deepcopy(env.bestnns.f)
    # make player
    player = MuPlayer(gspec, prediction_oracle, representation_oracle, dynamics_oracle,
                        mcts_params; S=Vector{Float32})

    function simulate_game(sim_id)
      # worker_sim_id += 1
      # Switch players' colors if necessary: "_pf" stands for "possibly flipped"
      if isa(player, TwoPlayers) && p.alternate_colors
        colors_flipped = sim_id % 2 == 1
        player_pf = colors_flipped ? flipped_colors(player) : player
      else
        colors_flipped = false
        player_pf = player
      end
      player_pf = player
      # Play the game and generate a report
      trace = play_game(gspec, player_pf, flip_probability=p.flip_probability)
      # report = simulator.measure(trace, colors_flipped, player)
      # Reset the player periodically
      # if !isnothing(p.reset_every) && worker_sim_id % p.reset_every == 0
      #   reset_player!(player)
      # end
      # Signal that a game has been simulated
      # game_simulated() 
      Threads.atomic_add!(total_simulated, 1) # don't know if there will be tension between adding, and getting number, but compiler should speedup it, so prevents it
      @logprogress total_simulated[] / p.num_games
      return trace
    end
    return (process = simulate_game, terminate = (() -> nothing))
  end
end

function self_play_step!(env::MuEnv)
  p = env.params.self_play_params.sim
  # @info Progress(1)
  traces = simulate(gspec, p)

  append!(env.memory, traces)
  @info "Self Play Step Finished"
end

function learning_step!(env::MuEnv)
  lp = env.params.learning_params
  ap = nothing

  #? symmetries 

  tr = MuTrainer(env.gspec, env.curnns, env.memory, env.params.learning_params, Flux.ADAM())
  nbatches = lp.batches_per_checkpoint

  @progress "learning step (epoch)" for _ in 1:lp.num_checkpoints
    update_weights!(tr, nbatches)
    if isnothing(ap)
      # env.curnns = tr.nns # trainer operate on shallow copy of curnns
      env.bestnns = deepcopy(env.curnns)
      @info "nns replaced" nns_replaced=true
    else
    end
  end
  @info "Learning Step finished"
end

function train!(env::MuEnv)
  while env.itc < env.params.num_iters
    @info "Starting Iteration" env.itc
    _, tsp = @timed self_play_step!(env) #TODO create custom time loggers
    _, tl = @timed learning_step!(env)
    @info "Iteration Finished" env.itc tsp tl
    env.itc += 1
  end
  @info "Training Finished"
end