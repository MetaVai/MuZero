using Base: AlwaysLockedST
function Benchmark.run(env::MuEnv, eval::AlphaZero.Benchmark.Evaluation)
  # net() = Network.copy(env.bestnns, on_gpu=eval.sim.use_gpu, test_mode=true)
  net() = deepcopy(env.bestnns)
  if isa(eval, Benchmark.Single)
    simulator = Simulator(net, record_trace) do net
      instantiate(eval.player, env.gspec, net)
    end
  else
    @assert isa(eval, Benchmark.Duel)
    simulator = Simulator(net, record_trace) do net
      player = Benchmark.instantiate(eval.player, env.gspec, net)
      baseline = Benchmark.instantiate(eval.baseline, env.gspec, net)
      return TwoPlayers(player, baseline)
    end
  end
  traces, elapsed = @timed simulate(
    simulator, env.gspec, eval.sim)
  # gamma = env.params.self_play.mcts.gamma
  # rewards, redundancy = rewards_and_redundancy(samples, gamma=gamma)
  lost, draw, won = count_wins(traces) ./ length(traces)
  @info "Benchmark" lost draw won time=elapsed
  return lost, draw, won
#   return Report.Evaluation(
#     name(eval), mean(rewards), redundancy, rewards, nothing, elapsed)
end

struct Mu <: Benchmark.Player
  params
end

# name(::Mu) = "MuZero"

function Benchmark.instantiate(p::Mu, gspec::AbstractGameSpec, nns)
  return MuPlayer(gspec, nns, p.params)
end

function run_duel(env, benchmark)
  report = []
  for duel in benchmark
    outcome = Benchmark.run(env, duel)
    push!(report, outcome)
  end
  return report
end
# benchmark
# env
# run_duel(env, benchmark)


# benchmark_sim = SimParams(
#     num_games=400,
#     num_workers=1,
#     batch_size=4,
#     use_gpu=false,
#     reset_every=1,
#     flip_probability=0.5,
#     alternate_colors=true)

# benchmark = [
#   Benchmark.Duel(
#     Mu(self_play.mcts),
#     Benchmark.MctsRollouts(self_play.mcts),
#     benchmark_sim)]


function count_wins(traces)
  results = (last(t.rewards) for t in traces)
  lost, draw, won = (count(==(i), results) for i in -1:1)
  @assert sum([lost, draw, won]) == length(results)
  # @info "Benchmark" lost draw won
  return lost, draw, won
end

