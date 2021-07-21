#####
##### Game trace
#####

"""
    MuTrace{State}

An object that collects all states visited during a game, along with the
rewards obtained at each step and the successive player policies to be used
as targets for the neural network.

# Constructor

    MuTrace(initial_state)

"""
struct MuTrace{State}
  states :: Vector{State}
  policies :: Vector{Vector{Float64}}
  rewards :: Vector{Float64}
  actions :: Vector{Int}
  rootvalues :: Vector{Float64}
  function MuTrace(init_state)
    return new{typeof(init_state)}([init_state], [], [], [], [])
  end
end

function valid_trace(t::MuTrace)
  return length(t.policies) == length(t.rewards) == length(t.actions) == length(t.states)-1
end

"""
    Base.push!(t::Trace, π, r, s, a, v)

Add a (target policy, reward, new state, selected action, rootvalue) to a trace.
"""
function Base.push!(t::MuTrace, π, r, s, a, v)
  push!(t.states, s)
  push!(t.policies, π)
  push!(t.rewards, r)
  push!(t.actions, a)
  push!(t.rootvalues, v)
end

function Base.length(t::MuTrace)
  return length(t.rewards)
end


function total_reward(t::MuTrace, gamma=1.)
  return sum(gamma^(i-1) * r for (i, r) in enumerate(t.rewards))
end

# function total_reward(t::Trace, gamma=1.)
#   return sum([gamma^(i-1) * r for (i, r) in enumerate(t.rewards)])
# end

# function debug_trace(gspec::AbstractGameSpec, t::Trace)
#   n = length(t)
#   for i in 1:n
#     println("Transition $i:")
#     game = GI.init(gspec, t.states[i])
#     GI.render(game)
#     for (a, p) in zip(GI.available_actions(game),  t.policies[i])
#       print("$(GI.action_string(gspec, a)): $(fmt(".3f", p))  ")
#     end
#     println("")
#     println("Obtained reward of: $(t.rewards[i]).")
#     println("")
#   end
#   println("Showing final state:")
#   GI.render(GI.init(gspec, t.states[n + 1]))
# end
