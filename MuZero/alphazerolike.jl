# These oracles mimics behaviour of AlphaZero

struct AlphaRepresentation
end

struct AlphaDynamics
  gspec 
end

struct AlphaPrediction
  net :: PredictionNetwork # PredictionNetwork
end

#AlphaZero-like behaviour 
function forward(representation_h::AlphaRepresentation, state)
  # hidden_state = Flux.flatten(state)
  hidden_state = state
  return hidden_state
end

function (representation_h::AlphaRepresentation)(state)
  # hidden_state = state
  hidden_state = forward(representation_h, state)
  return hidden_state
end

#AlphaZero-like behaviour 
function forward(dynamics_g::AlphaDynamics, hidden_state, action)
  game_g = GI.init(dynamics_g.gspec, hidden_state)
  GI.play!(game_g, action)
  reward = GI.white_reward(game_g)
  next_hidden_state = GI.current_state(game_g)
  return (reward, next_hidden_state)
end

function (dynamics_g::AlphaDynamics)(hidden_state, action)
  return forward(dynamics_g, hidden_state, action)
end

#AlphaZero-like behaviour
function forward(prediction_f::AlphaPrediction, hidden_state)
  x = Flux.flatten(hidden_state)
  forward(prediction_f.net, x)
end

function (prediction_f::AlphaPrediction)(hiddenstate)
  gspec = prediction_f.net.gspec
  x = GI.vectorize_state(gspec,hiddenstate)
  xnet = to_singletons(x)
  net_output = forward(prediction_f, xnet)
  p, v = from_singletons.(net_output)
  return (p, v[1])
end
Flux.@functor AlphaPrediction

# cache oracle (nn) results, so it can be quickly obtained without re-running network
struct CachedOracle{O, K, V}
  oracle :: O
  lookuptable :: Dict{K,V}
end

# oracle, typeof(State), typeof(Action), typeof(Reward) - used with dynamics
function CachedOracle(oracle, S, A=Int, R=Float64)
  return CachedOracle(
    oracle,
    Dict{Tuple{S,A}, Tuple{R,S}}()) # (sᵏ⁻¹,aᵏ) => (rᵏ,sᵏ)
end

function (oracle::CachedOracle)(args...)
  if haskey(oracle.lookuptable, args)
    answer = oracle.lookuptable[args]
  else
    answer = oracle.oracle(args...)
    oracle.lookuptable[args] = answer
  end
  return answer
end
