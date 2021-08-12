using Base: @kwdef, ident_cmp
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, flatten, relu, elu, softmax, unstack
import Zygote
using CUDA

to_singletons(x) = reshape(x, size(x)..., 1)
from_singletons(x) = reshape(x, size(x)[1:end-1])

abstract type AbstractPrediction end  # gspec, hyper, common, policyhead, valuehead
abstract type AbstractDynamics end    # gspec, hyper, common, statehead, rewardhead
abstract type AbstractRepresentation end # gspec, hyper, common

function forward(nn::AbstractRepresentation, observation)
  hiddenstate = nn.common(observation)
  return hiddenstate
end

function evaluate(nn::AbstractRepresentation, observation)
  x = GI.vectorize_state(nn.gspec, observation)
  # TODO: convert_input for GPU usage
  xnet = to_singletons(x)
  net_output = forward(nn, xnet)
  hiddenstate = from_singletons(net_output)
  #hiddenstate = net_output
  return hiddenstate
end

(nn::AbstractRepresentation)(observation) = evaluate(nn, observation)

function evaluate_batch(nn::AbstractRepresentation, batch)
  X = Flux.batch(GI.vectorize_state(nn.gspec, b) for b in batch)
  Xnet = to_nndevice(nn,X)
  net_outputs = forward(nn, Xnet)
  Hiddenstates = from_nndevice(nn, net_outputs)
  batchdim = ndims(Hiddenstates)
  return unstack(Hiddenstates, batchdim)
end



### Dynamics ###

function forward(nn::AbstractDynamics, hiddenstate_action)
  c = nn.common(hiddenstate_action)
  r = nn.rewardhead(c)
  s⁺¹ = nn.statehead(c)
  return (r, s⁺¹)
end

#TODO add GPU support
function evaluate(nn::AbstractDynamics, hiddenstate, action)
  snet = to_singletons(hiddenstate)
  batchdim = ndims(snet)
  if batchdim == 2
    avalactions = Base.OneTo(length(GI.actions(nn.gspec)))
    encoded_action = onehot(action, avalactions)
  else
    encoded_action = GI.encode_action(nn.gspec, action)
  end
  # gspec = nn.gspec
  # action_one_hot = onehot(action, GI.actions(gspec))
  anet = to_singletons(encoded_action)
  dim = ndims(snet)
  dim == 2 && (anet = Flux.flatten(anet))
  xnet = cat(snet, anet, dims=dim-1) # make dims=3 universal
  net_output = forward(nn, xnet)
  r, s = from_singletons.(net_output)
  return (r[1], s)
end

(nn::AbstractDynamics)((hiddenstate, action)) = evaluate(nn, hiddenstate, action)

function onehot(x::Integer, labels::Base.OneTo; type=Float32)
  result = zeros(type, length(labels))
  result[x] = one(type)
  return result
end

function encode_a(gspec, a; batchdim=4)
  if batchdim==2
    avalactions = Base.OneTo(length(GI.actions(gspec)))
    ret_a = onehot(a, avalactions)
  else
    ret_a = GI.encode_action(gspec, a)
  end
  return ret_a
end


function evaluate_batch(nn::AbstractDynamics, batch)
  S = Flux.batch(b[1] for b in batch)
  batchdim = ndims(S)
  A = Flux.batch(encode_a(nn.gspec, b[2]; batchdim) for b in batch)
  batchdim == 2 && (A = Flux.flatten(A))
  X = cat(S, A, dims=batchdim-1)
  Xnet = to_nndevice(nn,X)
  net_outputs = forward(nn, Xnet)
  (R, S⁺¹) = from_nndevice(nn,net_outputs)
  # # 
  # R_itr = unstack_itr(R, 2)
  # S⁺¹_itr = unstack_itr(S⁺¹, batchdim)
  # return collect(zip(R_itr, S⁺¹_itr))
  # return [(R[1,i], S⁺¹[:,i]) for i in eachindex(batch)]
  # return [(R[1,i], S⁺¹[:,:,:,i]) for i in eachindex(batch)]
  # return [(R[1,i], collect(selectdim(S⁺¹,batchdim,i))) for i in eachindex(batch)]
  return collect(zip(unstack(R,2), unstack(S⁺¹,batchdim)))
end


### Prediction ###

function forward(nn::AbstractPrediction, hiddenstate)
  c = nn.common(hiddenstate)
  v = nn.valuehead(c)
  p = nn.policyhead(c)
  return (p, v)
end

function evaluate(nn::AbstractPrediction, hiddenstate)
  x = hiddenstate
  xnet = to_singletons(x)
  net_output = forward(nn, xnet)
  p, v = from_singletons.(net_output)
  return (p, v[1])
end

(nn::AbstractPrediction)(hiddenstate) = evaluate(nn, hiddenstate)

function evaluate_batch(nn::AbstractPrediction, batch)
  X = Flux.batch(batch)
  Xnet = to_nndevice(nn, X)
  net_output = forward(nn, Xnet)
  P, V = from_nndevice(nn, net_output)
  return collect(zip(unstack(P,2), unstack(V,2)))
end


# TODO create constructor from gspec, hidden_state_shape... and get rid of gspecs
struct MuNetworkHP{GameSpec, Fhp, Ghp, Hhp}
  # hidden_state_shape
  gspec :: GameSpec
  predictionHP :: Fhp
  dynamicsHP :: Ghp
  representationHP :: Hhp
end

struct MuNetwork{F<:AbstractPrediction,G<:AbstractDynamics,H<:AbstractRepresentation}
  params :: MuNetworkHP
  f :: F
  g :: G
  h :: H
end

function MuNetwork(params::MuNetworkHP)
  fHP = params.predictionHP
  gHP = params.dynamicsHP
  hHP = params.representationHP
  # @assert fHP.hiddenstate_shape == gHP.hiddenstate_shape == hHP.hiddenstate_shape
  f = PredictionNetwork(params.gspec, fHP)
  g = DynamicsNetwork(params.gspec, gHP)
  h = RepresentationNetwork(params.gspec, hHP)
  return MuNetwork(params, f, g, h)
end

# takes output form neural netrork back to CPU, and unstack it along last dimmension
function convert_output(X)
  X = from_nndevice(nothing, X)
  return Flux.unstack(X, ndims(X))
end

struct InitialOracle{H<:AbstractRepresentation, F<:AbstractPrediction}
  h :: H
  f :: F
end

InitialOracle(nns::MuNetwork) = InitialOracle(nns.h, nns.f)

(init::InitialOracle)(observation) = evaluate(init, observation)

# function evaluate_batch(init::InitialOracle, batch)
#   X = Flux.batch(GI.vectorize_state(init.f.gspec, b) for b in batch) #obsrvation
#   Xnet = to_nndevice(init.f, X)
#   S⁰ = forward(init.h, Xnet) # hiddenstate
#   P⁰, V⁰ = forward(init.f, S⁰) # policy, value

#   P⁰, V⁰, S⁰ = map(convert_output, (P⁰, V⁰, S⁰))
#   V⁰ = [v[1] for v in V⁰]
#   return collect(zip(P⁰, V⁰, S⁰, zero(V⁰)))
# end
function evaluate_batch(init::InitialOracle, batch)
  X = Flux.batch(GI.vectorize_state(init.f.gspec, b) for b in batch) #obsrvation
  Xnet = to_nndevice(init.f, X)
  S⁰ = forward(init.h, Xnet) # hiddenstate
  P⁰, V⁰ = forward(init.f, S⁰) # policy, value
  P⁰, V⁰, S⁰ = map(convert_output, (P⁰, V⁰, S⁰))
  V⁰ = [v[1] for v in V⁰]
  return collect(zip(P⁰, V⁰, S⁰, zero(V⁰)))
end

#TODO test nonmutable struct
struct RecurrentOracle{G<:AbstractDynamics, F<:AbstractPrediction}
  g :: G
  f :: F
end

RecurrentOracle(nns::MuNetwork) = RecurrentOracle(nns.g, nns.f)

(recur::RecurrentOracle)((state,action)) = evaluate(recur, (state,action))

function evaluate_batch(recur::RecurrentOracle, batch)
  S = Flux.batch(b[1] for b in batch)
  batchdim = ndims(S)
  A = Flux.batch(encode_a(recur.f.gspec, b[2]; batchdim) for b in batch)
  S_A = cat(S,A,dims=batchdim-1)
  S_A_net = to_nndevice(recur.f, S_A) # assuming all networks are on the same device
  R, S⁺¹ = forward(recur.g, S_A_net)
  P⁺¹, V⁺¹ = forward(recur.f, S⁺¹)
  P⁺¹, V⁺¹, R, S⁺¹ = map(convert_output, (P⁺¹, V⁺¹, R, S⁺¹))
  V⁺¹ = (v[1] for v in V⁺¹)
  R = (r[1] for r in R)
  return collect(zip(P⁺¹, V⁺¹, S⁺¹, R)) #TODO check for memory consumption;s implement this everywhere,
end
 # TODO cleanup the rest
evaluate(nn, x) = evaluate_batch(nn, [x])[1]

# TODO regularized params
regularized_params_(l) = []
regularized_params_(l::Flux.Dense) = [l.weight]
regularized_params_(l::Flux.Conv) = [l.weight]

function regularized_params(net::MuNetwork)
  return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

#adhoc
function regularized_params(net)
  return (w for l in Flux.modules(net.f.net) for w in regularized_params_(l))
end


array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

include("./architectures/simplemlp.jl")
include("./architectures/resnet.jl")

#TODO this it kinda ugly, maybe create macro that create proper on_gpu based on networks
on_gpu(nn::Union{PredictionSimpleNetwork,PredictionResnetNetwork}) = array_on_gpu(nn.valuehead[end].bias)
on_gpu(nn::Union{DynamicsSimpleNetwork,DynamicsResnetNetwork}) = array_on_gpu(nn.rewardhead[end].bias)
on_gpu(nn::RepresentationSimpleNetwork) = array_on_gpu(nn.common[end].bias)
on_gpu(nn::RepresentationResnetNetwork) = array_on_gpu(nn.common[end][1].layers[1].weight)

to_nndevice(nn, x) = on_gpu(nn) ? Flux.gpu(x) : x
from_nndevice(nn, x) = Flux.cpu(x)

# remembert to put fields in correct order, otherwise they are swapped when |>gpu
Flux.@functor RepresentationSimpleNetwork (common,)
Flux.@functor RepresentationResnetNetwork (common,)
Flux.@functor DynamicsSimpleNetwork       (common, statehead, rewardhead)
Flux.@functor DynamicsResnetNetwork       (common, statehead, rewardhead)
Flux.@functor PredictionSimpleNetwork     (common, policyhead, valuehead)
Flux.@functor PredictionResnetNetwork     (common, policyhead, valuehead)
Flux.@functor MuNetwork                   (f, g, h)
