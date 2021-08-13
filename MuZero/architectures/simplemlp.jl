##### Representation #####

@kwdef struct RepresentationSimpleHP
  width :: Int
  depth :: Int
  hiddenstate_shape :: Int
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct RepresentationSimpleNetwork <: AbstractRepresentation
  gspec
  hyper
  common
end

function RepresentationNetwork(gspec::AbstractGameSpec, hyper::RepresentationSimpleHP)
  bnmom = hyper.batch_norm_momentum
  function make_dense(indim, outdim)
    if hyper.use_batch_norm
      Chain(
      Dense(indim, outdim, bias=false),
      BatchNorm(outdim, relu, momentum=bnmom))
    else
      Dense(indim, outdim, relu)
    end
  end
  indim = prod(GI.state_dim(gspec))
  outdim = hyper.hiddenstate_shape
  hsize = hyper.width
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
  if hyper.depth == -1 # somewhat unintuitive, jump from 1 to 3 layers #? depth-1 
    common = Chain(
      flatten,
      # make_dense(indim, outdim)
      Dense(indim, outdim)
    )
  else
    common = Chain(
      flatten,
      make_dense(indim, hsize),
      hlayers(hyper.depth)...,
      make_dense(hsize, outdim)
    )
  end
  RepresentationSimpleNetwork(gspec, hyper, common)
end

### general SimpleNet used in Prediction and Dynamics
# TODO make constructor from vector of sizes 
@kwdef struct SimpleNetHP_
  indim :: Int
  outdim :: Int
  width :: Int
  depth_common :: Int
  depth_vectorhead :: Int = 1
  depth_scalarhead :: Int = 1
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct SimpleNet_
  gspec
  hyper
  common
  vectorhead
  scalarhead
end
# TODO remove gspec dependence
function SimpleNet_(gspec::AbstractGameSpec, hyper::SimpleNetHP_)
  bnmom = hyper.batch_norm_momentum
  function make_dense(indim, outdim)
    if hyper.use_batch_norm
      Chain(
      Dense(indim, outdim, bias=false),
      BatchNorm(outdim, elu, momentum=bnmom))
    else
      Dense(indim, outdim, elu)
    end
  end
  indim = hyper.indim
  outdim = hyper.outdim
  hsize = hyper.width
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth] #? 1:depth-1
  # common = depth_common == -1 ?
  #   flatten :
  #   Chain(
  #     flatten, #?
  #     make_dense(indim, hsize),
  #     hlayers(hyper.depth_common)...)
  # scalarhead = Chain(
  #   hlayers(hyper.depth_scalarhead)...,
  #   Dense(hsize, 1, tanh))
  # vectorhead = Chain(
  #   hlayers(hyper.depth_vectorhead)...,
  #   Dense(hsize, outdim),
  #   softmax)
  if hyper.depth_common == -1
    common = identity
    outcomm = indim
  else
    common = Chain(
      flatten, #? identity
      make_dense(indim, hsize),
      hlayers(hyper.depth_common)...)
    outcomm = hsize
  end
  if hyper.depth_scalarhead == -1
    scalarhead = Dense(outcomm, 1, tanh)
  else
    scalarhead = Chain(
      outcomm != hsize ? make_dense(outcomm, hsize) : identity,
      hlayers(hyper.depth_scalarhead)...,
      Dense(hsize, 1, tanh))
  end
  if hyper.depth_vectorhead == -1
    vectorhead = Chain(Dense(outcomm, outdim))
  else
    vectorhead = Chain(
      outcomm != hsize ? make_dense(outcomm, hsize) : identity,
      hlayers(hyper.depth_vectorhead)...,
      Dense(hsize, outdim)
      )
  end

  SimpleNet_(gspec, hyper, common, vectorhead, scalarhead)
end


### Dynamics ###

@kwdef struct DynamicsSimpleHP
  hiddenstate_shape :: Int
  width :: Int
  depth_common :: Int
  depth_vectorhead :: Int = 1 # depth state-head
  depth_scalarhead :: Int = 1 # depth reward-head
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct DynamicsSimpleNetwork <: AbstractDynamics
  gspec
  hyper
  common
  statehead
  rewardhead
end

function DynamicsNetwork(gspec::AbstractGameSpec, hyper::DynamicsSimpleHP)
  indim = hyper.hiddenstate_shape + GI.num_actions(gspec)
  outdim = hyper.hiddenstate_shape
  simplenethyper = SimpleNetHP_(indim,
    outdim,
    hyper.width,
    hyper.depth_common,
    hyper.depth_vectorhead,
    hyper.depth_scalarhead,
    hyper.use_batch_norm,
    hyper.batch_norm_momentum)
  simplenet = SimpleNet_(gspec,simplenethyper)
  DynamicsSimpleNetwork(gspec, hyper, simplenet.common, simplenet.vectorhead, simplenet.scalarhead)
end


### Prediction ###

@kwdef struct PredictionSimpleHP
  hiddenstate_shape :: Int
  width :: Int
  depth_common :: Int
  depth_vectorhead :: Int = 1 # depth policy-head
  depth_scalarhead :: Int = 1 # depth value-head
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct PredictionSimpleNetwork <: AbstractPrediction
  gspec
  hyper
  common
  policyhead
  valuehead
end

function PredictionNetwork(gspec::AbstractGameSpec, hyper::PredictionSimpleHP)
  indim = hyper.hiddenstate_shape
  outdim = GI.num_actions(gspec)
  simplenethyper = SimpleNetHP_(indim,
    outdim,
    hyper.width,
    hyper.depth_common,
    hyper.depth_vectorhead,
    hyper.depth_scalarhead,
    hyper.use_batch_norm,
    hyper.batch_norm_momentum)
  simplenet = SimpleNet_(gspec,simplenethyper)
  PredictionSimpleNetwork(gspec, hyper, simplenet.common, Chain(simplenet.vectorhead, softmax), simplenet.scalarhead)
end

# function evaluate_batch(nn::AbstractPrediction, batch) # is the same as resnet
