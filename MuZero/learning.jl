using Statistics: mean
import Zygote
# using StatsBase: mean, weights
# import StatsBase
# choose trace, and game pos from uniform distribution
sample_trace(memory) = rand(memory) #!
# sample_trace(memory) = memory[1]
sample_position(trace) = rand(1:length(trace)) #! HERE
# sample_position(trace) = 2

# TODO comments and documentation
#! wp - is white playing
# TODO debug indexes (sᵏ⁻¹, aᵏ) are index 1
function make_target(gspec, trace, state_idx, hyper)
  Ksteps = hyper.num_unroll_steps
  td_steps = hyper.td_steps
  γ = hyper.discount

  observation = trace.states[state_idx]
  x = GI.vectorize_state(gspec, observation) # convert observation to h_nn input
  a_mask = GI.actions_mask(GI.init(gspec, observation))
  wp = observation.curplayer
  # TODO change into staticvector or smth fast - sizehint!
  target_values   = Float64[]         ; sizehint!(target_values, Ksteps+1)
  target_rewards  = Float64[]         ; sizehint!(target_rewards, Ksteps+1)
  target_policies = Vector{Float64}[] ; sizehint!(target_policies, Ksteps+1)
  #? should it be 2 dim array for easy f32 conversion
  #//target_policies = Array{Float64}(undef, 9, 0)
  actions         = Int[]

  #? change to map
  for idx in state_idx:state_idx+Ksteps
    # The value target is the discounted root value of the search tree td_steps into the
    # future, plus the discounted sum of all rewards until then.
    bootstrap_idx = idx + td_steps
    if bootstrap_idx <= length(trace)
      value = trace.rootvalues[bootstrap_idx] * γ^td_steps
    else
      bootstrap_idx = length(trace)
      value = 0
    end

    for (i, reward) in enumerate(trace.rewards[idx:bootstrap_idx])
      value += reward * γ^(i-1) # numerating from 1
    end

    #= google implementation use masks for every target, ex:
      reward_mask = 1.0 if current_index > state_index else 0.0 =#
    if idx <= length(trace)
      reward = trace.rewards[idx]
      if GI.two_players(gspec)
        if !wp
          value = -value
          reward = -reward #? wp ? reward : -reward
        end
        wp = !wp
      end  
      push!(target_values, value)
      push!(target_rewards, reward)
      # create full length policy vector
      a_m = GI.actions_mask(GI.init(gspec, trace.states[idx]))
      p = zeros(size(a_m))
      p[a_m] = trace.policies[idx]
      push!(target_policies, p)
      #target_policies = [target_policies p]
      push!(actions, trace.actions[idx])
    else
      push!(target_values, 0)
      push!(target_rewards, 0)

      uniform_policy = [1/length(trace.policies[1]) for _ in 1:length(trace.policies[1])]
      push!(target_policies, uniform_policy)
      #target_policies = [target_policies uniform_policy]
      push!(actions, rand(1:length(trace.policies[1])))
    end
  end
  return (; x, a_mask, as=actions, vs=target_values, rs=target_rewards, ps=reduce(hcat,target_policies))
end

function sample_batch(gspec, memory, hyper)
  traces = [sample_trace(memory) for _ in 1:hyper.loss_computation_batch_size]
  trace_pos_idxs = [sample_position(t) for t in traces] #! ERROR; DO NOT USE GENERATORS WITH RAND()
  samples = (make_target(gspec, t, i, hyper) for (t,i) in zip(traces, trace_pos_idxs))

  X       = Flux.batch(smpl.x       for smpl in samples)
  A_mask  = Flux.batch(smpl.a_mask  for smpl in samples)
  As      = Flux.batch(smpl.as      for smpl in samples)
  Ps      = Flux.batch(smpl.ps      for smpl in samples)
  Vs      = Flux.batch(smpl.vs      for smpl in samples)
  Rs      = Flux.batch(smpl.rs      for smpl in samples)
  f32(arr) = convert(AbstractArray{Float32}, arr)
  return map(f32, (; X, A_mask, As, Ps, Vs, Rs))
end

lossₚ(p̂, p)::Float32 = Flux.Losses.crossentropy(p̂, p) #TODO move to hyper
lossᵥ(v̂, v)::Float32 = Flux.Losses.mse(v̂, v)
lossᵣ(r̂, r)::Float32 = 0f0

# TODO add assertions about sizes
function losses(nns, hyper, (X, A_mask, As, Ps, Vs, Rs))
  prediction, dynamics, representation = nns.f, nns.g, nns.h
  creg::Float32 = hyper.l2_regularization
  Ksteps = hyper.num_unroll_steps

  # initial step, from the real observation
  Hiddenstate = forward(representation, X)
  P̂⁰, V̂⁰ = forward(prediction, Hiddenstate)
  P̂⁰ = normalize_p(P̂⁰, A_mask)
  # R̂⁰ = zero(V̂⁰)

  Lp = lossₚ(P̂⁰, Ps[ : ,1,:]) # scale=1
  Lv = lossᵥ(V̂⁰, Vs[1:1,:])
  Lr = zero(Lv) # starts at next step (see MuZero paper appendix)
  
  # recurrent inference 
  for i in 2:Ksteps+1
    A = As[i-1,:] 
    R̂, Hiddenstate = forward(dynamics, Hiddenstate, A) # obtain next hiddenstate
    P̂, V̂ = forward(prediction, Hiddenstate) #? should flip V based on players
    # scale loss so that the overall weighting of the recurrent_inference (g,f nns)
    # is equal to that of the initial_inference (h,f nns)
    scale = 1f0 / Ksteps #? instead of constant scale, maybe 2^(-i+1)
    Lp += lossₚ(P̂, Ps[ : ,i,:]) * scale #? @view
    Lv += lossᵥ(V̂, Vs[i:i,:])   * scale 
    Lr += lossᵣ(R̂, Rs[i:i,:])   * scale
  end
  Lreg = iszero(creg) ? zero(Lv) : creg * sum(sum(w.^2) for w in regularized_params(nns))
  L = Lp + Lv + Lreg # + Lr
  # L = Lp + Lreg # + Lr
  Zygote.@ignore @debug "Loss" loss_total=L loss_policy=Lp loss_value=Lv loss_reg_params=Lreg #? check if compute means inside logger is avaliable
  return (L, Lp, Lv, Lr, Lreg)
end

#TODO replace Zygote.withgradient() - new version
function lossgrads(f, args...)
  val, back = Zygote.pullback(f, args...)
  grad = back(Zygote.sensitivity(val))
  return val, grad
end
  

# function train!(nns, opt, loss, data; cb=()->())
function μtrain!(nns, loss, data, opt)
  ps = Flux.params(nns)
  losses = Float32[]
  # @progress "learning step (checkpoint)" for (i, d) in enumerate(data)
  for (i, d) in enumerate(data)
    l, gs = lossgrads(ps) do
      loss(d...)
    end
    push!(losses, l)
    Flux.update!(opt, ps, gs)
  end
  @info "Loss" mean_loss_total = mean(losses)
end

struct MuTrainer{Network,M}
  gspec
  nns :: Network # MuNetwork
  memory :: M #? don't know if memory pointer in trainter is good idea
  hyper
  opt #TODO create opt every update_weights!() from opt_recipe
end

function update_weights!(tr::MuTrainer, n)
  L(batch...) = losses(tr.nns, tr.hyper, batch)[1]
  #? move computing samples into Trainer constructor
  samples = (sample_batch(tr.gspec, tr.memory, tr.hyper) for _ in 1:n)
  # opt = tr.hyper.opt_recipe()
  μtrain!(tr.nns, L, samples, tr.opt)
  #? GC 
end