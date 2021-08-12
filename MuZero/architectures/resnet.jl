using Flux: Conv, SkipConnection, relu
"""
    ResNetHP

Hyperparameters for the convolutional resnet architecture.

| Parameter                 | Type                | Default   |
|:--------------------------|:--------------------|:----------|
| `num_blocks`              | `Int`               |  -        |
| `num_filters`             | `Int`               |  -        |
| `conv_kernel_size`        | `Tuple{Int, Int}`   |  -        |
| `num_policy_head_filters` | `Int`               | `2`       |
| `num_value_head_filters`  | `Int`               | `1`       |
| `batch_norm_momentum`     | `Float32`           | `0.6f0`   |

The trunk of the two-head network consists of `num_blocks` consecutive blocks.
Each block features two convolutional layers with `num_filters` filters and
with kernel size `conv_kernel_size`. Note that both kernel dimensions must be
odd.

During training, the network is evaluated in training mode on the whole
dataset to compute the loss before it is switched to test model, using
big batches. Therefore, it makes sense to use a high batch norm momentum
(put a lot of weight on the latest measurement).

# AlphaGo Zero Parameters

The network in the original paper from Deepmind features 20 blocks with 256
filters per convolutional layer.
"""
@kwdef struct ResNetHP_
  indim :: Tuple{Int, Int, Int}
  outdim_head1 :: Int
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  num_head1_filters :: Int = 2 # policy
  num_head2_filters :: Int = 1 # value
  batch_norm_momentum :: Float32 = 0.6f0
end

"""
    ResNet <: TwoHeadNetwork

The convolutional residual network architecture that is used
in the original AlphaGo Zero paper.
"""
mutable struct ResNet_
  gspec
  hyper
  common
  head1 #policy/state (vector/tensor like)
  head2 #value/reward (scalar like)
end

function ResNetBlock(size, n, bnmom)
  pad = size .÷ 2
  layers = Chain(
    Conv(size, n=>n, pad=pad, bias=false),
    BatchNorm(n, relu, momentum=bnmom),
    Conv(size, n=>n, pad=pad, bias=false),
    BatchNorm(n, momentum=bnmom))
  return Chain(
    SkipConnection(layers, +),
    x -> relu.(x))
    # relu)
end

function ResNet_(gspec::AbstractGameSpec, hyper::ResNetHP_)
  indim = hyper.indim
  outdim = hyper.outdim_head1
  ksize = hyper.conv_kernel_size
  @assert all(ksize .% 2 .== 1)
  pad = ksize .÷ 2
  nf = hyper.num_filters
  npf = hyper.num_head1_filters
  nvf = hyper.num_head2_filters
  bnmom = hyper.batch_norm_momentum
  common = Chain(
    Conv(ksize, indim[3]=>nf, pad=pad, bias=false),
    BatchNorm(nf, relu, momentum=bnmom),
    [ResNetBlock(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
  phead = Chain(
    Conv((1, 1), nf=>npf, bias=false),
    BatchNorm(npf, relu, momentum=bnmom),
    flatten,
    Dense(indim[1] * indim[2] * npf, outdim),
    softmax)
  vhead = Chain(
    Conv((1, 1), nf=>nvf, bias=false),
    BatchNorm(nvf, relu, momentum=bnmom),
    flatten,
    Dense(indim[1] * indim[2] * nvf, nf, relu),
    Dense(nf, 1, tanh))
  ResNet_(gspec, hyper, common, phead, vhead)
end

### Representation ###

@kwdef struct RepresentationResnetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct RepresentationResnetNetwork <: AbstractRepresentation
  gspec
  hyper
  common
end

function RepresentationNetwork(gspec::AbstractGameSpec, hyper::RepresentationResnetHP)
  indim = GI.state_dim(gspec)
  resnethyper = ResNetHP_(
    indim,
    1, # outdim not used
    hyper.num_blocks,
    hyper.num_filters,
    hyper.conv_kernel_size,
    1, # num_head1_filters not used
    1, # num_head2_filters not used
    hyper.batch_norm_momentum
  )
  resnet = ResNet_(gspec, resnethyper)
  common = Chain(resnet.common,
    Conv((1,1), hyper.num_filters=>indim[3], relu)
  )
  return RepresentationResnetNetwork(gspec, hyper, common)
end


### Dynamics ###

@kwdef struct DynamicsResnetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  num_reward_head2_filters :: Int = 1 # head2
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct DynamicsResnetNetwork <: AbstractDynamics
  gspec
  hyper
  common
  statehead
  rewardhead
end

# Dynamics Network with identity as state head
function DynamicsNetwork(gspec::AbstractGameSpec, hyper::DynamicsResnetHP)
  state_dim = GI.state_dim(gspec)
  # indim = (state_dim[1], state_dim[2], hyper.num_filters+1)
  indim = (state_dim[1], state_dim[2], state_dim[3]+1)
  resnethyper = ResNetHP_(
    indim, # indim not used
    1, # outdim_head1 not used
    hyper.num_blocks,
    hyper.num_filters,
    hyper.conv_kernel_size,
    1, # num_head1_filters not used
    hyper.num_reward_head2_filters,
    hyper.batch_norm_momentum
  )
  resnet = ResNet_(gspec, resnethyper)
  small_state = Conv((1,1), hyper.num_filters=>state_dim[3], relu) # a lot quicker hash
  return DynamicsResnetNetwork(gspec, hyper,
    resnet.common,
    small_state, # state head as identity
    resnet.head2)
end

@kwdef struct PredictionResnetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  num_policy_head1_filters :: Int = 2 # head1
  num_value_head2_filters :: Int = 1 # head2
  batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct PredictionResnetNetwork <: AbstractPrediction
  gspec
  hyper
  common
  policyhead
  valuehead
end

function PredictionNetwork(gspec::AbstractGameSpec, hyper::PredictionResnetHP)
  indim = GI.state_dim(gspec)
  outdim_head1 = GI.num_actions(gspec)
  resnethyper = ResNetHP_(
    indim, # indim not used
    outdim_head1,
    hyper.num_blocks,
    hyper.num_filters,
    hyper.conv_kernel_size,
    hyper.num_policy_head1_filters,
    hyper.num_value_head2_filters,
    hyper.batch_norm_momentum
  )
  resnet = ResNet_(gspec, resnethyper)
  return PredictionResnetNetwork(gspec, hyper,
    resnet.common, # leaving only residual layers
    resnet.head1,
    resnet.head2)
end