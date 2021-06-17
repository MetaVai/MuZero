using Base: @kwdef
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, flatten, relu, softmax, onehot

# TODO create constructors from MuNetworkHP
struct MuNetworkHP
    hidden_state_shape
    predictionHP
    dynamicsHP
    representationHP
end

# function ResNetBlock(size, n, bnmom)
#     pad = size .÷ 2
#     layers = Chain(
#         Conv(size, n=>n, pad=pad),
#         BatchNorm(n, relu, momentum=bnmom),
#         Conv(size, n=>n, pad=pad),
#         BatchNorm(n, momentum=bnmom))
#     return Chain(
#         SkipConnection(layers, +),
#         x -> relu.(x))
# end

to_singletons(x) = reshape(x, size(x)..., 1)
from_singletons(x) = reshape(x, size(x)[1:end-1])

##### Representation #####

@kwdef struct RepresentationHP
    width :: Int
    depth :: Int
    hiddenstate_shape :: Int
    use_batch_norm :: Bool = false
    batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct RepresentationNetwork
    gspec
    hyper
    architecture
end

function RepresentationNetwork(gspec::AbstractGameSpec, hyper::RepresentationHP)
    bnmom = hyper.batch_norm_momentum
    function make_dense(indim, outdim)
        if hyper.use_batch_norm
            Chain(
            Dense(indim, outdim),
            BatchNorm(outdim, relu, momentum=bnmom))
        else
            Dense(indim, outdim, relu)
        end
    end
    indim = prod(GI.state_dim(gspec))
    outdim = hyper.hiddenstate_shape
    hsize = hyper.width
    hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
    architecture = Chain(
        flatten,
        make_dense(indim, hsize),
        hlayers(hyper.depth)...,
        make_dense(hsize, outdim)
    )
    RepresentationNetwork(gspec, hyper, architecture)
end

function forward(nn::RepresentationNetwork, observation)
    hiddenstate = nn.architecture(observation)
    return hiddenstate
end

function evaluate(nn::RepresentationNetwork, observation)
    gspec = nn.gspec
    x = GI.vectorize_state(gspec, observation)
    # TODO: convert_input for GPU usage
    xnet = to_singletons(x)
    net_output = forward(nn, xnet)
    hiddenstate = from_singletons(net_output)
    #hiddenstate = net_output
    return hiddenstate
end

(nn::RepresentationNetwork)(observation) = evaluate(nn, observation)



### general SimpleNet

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
    scalarhead
    vectorhead
end

function SimpleNet_(gspec::AbstractGameSpec, hyper::SimpleNetHP_)
    bnmom = hyper.batch_norm_momentum
    function make_dense(indim, outdim)
        if hyper.use_batch_norm
            Chain(
            Dense(indim, outdim),
            BatchNorm(outdim, relu, momentum=bnmom))
        else
            Dense(indim, outdim, relu)
        end
    end
    indim = hyper.indim
    outdim = hyper.outdim
    hsize = hyper.width
    hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
    common = Chain(
        flatten,
        make_dense(indim, hsize),
        hlayers(hyper.depth_common)...)
    scalarhead = Chain(
        hlayers(hyper.depth_scalarhead)...,
        Dense(hsize, 1, tanh))
    vectorhead = Chain(
        hlayers(hyper.depth_vectorhead)...,
        Dense(hsize, outdim),
        softmax)
    SimpleNet_(gspec, hyper, common, scalarhead, vectorhead)
end


### Dynamics ###

@kwdef struct DynamicsHP
    hiddenstate_shape :: Int
    width :: Int
    depth_common :: Int
    depth_vectorhead :: Int = 1
    depth_scalarhead :: Int = 1
    use_batch_norm :: Bool = false
    batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct DynamicsNetwork
    gspec
    hyper
    common
    scalarhead
    vectorhead
end

function DynamicsNetwork(gspec::AbstractGameSpec, hyper::DynamicsHP)
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
    DynamicsNetwork(gspec, hyper, simplenet.common, simplenet.scalarhead, simplenet.vectorhead)
end

function forward(nn::DynamicsNetwork, hiddenstate_action)
    c = nn.common(hiddenstate_action)
    r = nn.scalarhead(c)
    s₊₁ = nn.vectorhead(c)
    return (r, s₊₁)
end

function evaluate(nn::DynamicsNetwork, hiddenstate, action)
    gspec = nn.gspec
    action_one_hot = onehot(action, GI.actions(gspec))
    x = cat(hiddenstate, action_one_hot, dims=1)
    xnet = to_singletons(x)
    net_output = forward(nn, xnet)
    r, s = from_singletons.(net_output)
    return (r[1], s)
end

(nn::DynamicsNetwork)(hiddenstate,action) = evaluate(nn,hiddenstate,action)

### Prediction ###

@kwdef struct PredictionHP
    hiddenstate_shape :: Int
    width :: Int
    depth_common :: Int
    depth_vectorhead :: Int = 1
    depth_scalarhead :: Int = 1
    use_batch_norm :: Bool = false
    batch_norm_momentum :: Float32 = 0.6f0
end

mutable struct PredictionNetwork
    gspec
    hyper
    common
    scalarhead
    vectorhead
end

function PredictionNetwork(gspec::AbstractGameSpec, hyper::PredictionHP)
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
    PredictionNetwork(gspec, hyper, simplenet.common, simplenet.scalarhead, simplenet.vectorhead)
end

function forward(nn::PredictionNetwork, hiddenstate_action)
    c = nn.common(hiddenstate_action)
    v = nn.scalarhead(c)
    p = nn.vectorhead(c)
    return (p, v)
end

function evaluate(nn::PredictionNetwork, hiddenstate)
    x = hiddenstate
    xnet = to_singletons(x)
    net_output = forward(nn, xnet)
    p, v = from_singletons.(net_output)
    return (p, v[1])
end

(nn::PredictionNetwork)(hiddenstate) = evaluate(nn, hiddenstate)

# ###
# netparams = NetLib.SimpleNetHP( # tictactoe
#     width=200,
#     depth_common=6,
#     use_batch_norm=true,
#     batch_norm_momentum=1.)

# netparams = NetLib.ResNetHP( # connect four
#     num_filters=128,
#     num_blocks=5,
#     conv_kernel_size=(3, 3),
#     num_policy_head_filters=32,
#     num_value_head_filters=32,
#     batch_norm_momentum=0.1)
