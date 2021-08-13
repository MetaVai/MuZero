# using Base: AlwaysLockedST
function Benchmark.run(env::MuEnv, eval::AlphaZero.Benchmark.Evaluation)
  # net() = Network.copy(env.bestnns, on_gpu=eval.sim.use_gpu, test_mode=true)
  device = eval.sim.use_gpu ? Flux.gpu : Flux.cpu
  function net()
    nns = deepcopy(env.bestnns) |> device |> Flux.testmode!
    return (InitialOracle(nns), RecurrentOracle(nns))
  end
  if isa(eval, Benchmark.Single)
    simulator = Simulator(net, record_trace) do net
      Benchmark.instantiate(eval.player, env.gspec, net)
    end
  else
    @assert isa(eval, Benchmark.Duel)
    simulator = Simulator(net, record_trace) do net
      player = Benchmark.instantiate(eval.player, env.gspec, net)
      baseline = Benchmark.instantiate(eval.baseline, env.gspec, net)
      return TwoPlayers(player, baseline)
    end
  end
  samples, elapsed = @timed simulate(
    simulator, env.gspec, eval.sim)
  # gamma = env.params.self_play.mcts.gamma
  rewards, redundancy = rewards_and_redundancy(samples, gamma=1.0)
  opponent_won, draw, mu_won = count_wins(rewards) ./ length(rewards)
  # @info "Benchmark" white_lost draw white_won time=elapsed
  return (; opponent_won, draw, mu_won, time_benchmarked=elapsed)
#   return Report.Evaluation(
#     name(eval), mean(rewards), redundancy, rewards, nothing, elapsed)
end

struct Mu <: Benchmark.Player
  params :: MctsParams
  # make_oracles # Function make_oracles()
end

# name(::Mu) = "MuZero"

function Benchmark.instantiate(p::Mu, gspec::AbstractGameSpec, nns)
  return MuPlayer(nns, p.params)
end

function run_duel(env, benchmark)
  # report = []
  report = Dict()
  for (name,duel) in pairs(benchmark)
    outcome = Benchmark.run(env, duel)
    # push!(report, (;name=outcome))
    report[name] = outcome
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


function count_wins(rewards)
  lost, draw, won = (count(==(i), rewards) for i in -1:1)
  @assert sum([lost, draw, won]) == length(rewards)
  # @info "Benchmark" lost draw won
  return lost, draw, won
end

