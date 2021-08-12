include("./script_resnet.jl")
##

using BenchmarkTools
using Flux: cpu, gpu

@profview self_play_step!(envsim)
@enter self_play_step!(envres)
@timed self_play_step!(envsim)
@timed self_play_step!(envres)
@timed learning_step!(envres)
@profview learning_step!(envres)
@profview self_play_step!(envres)


game = GI.init(gspec)
nns = envres.bestnns
player = MuPlayer((InitialOracle(nns), RecurrentOracle(nns)), envres.params.self_play.mcts)

@enter AlphaZero.think(player, game)
AlphaZero.think(player, game)
@profview for i in 1:100; AlphaZero.think(player, game); end;

batchsize = 8
actionbatch = [rand(1:5) for _ in 1:batchsize]

obsbatch = []
for a in actionbatch
  GI.play!(game, a)
  obs = GI.current_state(game)
  push!(obsbatch, obs)
end

obsbatch

statebatchres = evaluate_batch(envres.bestnns.h, obsbatch)
statebatchsim = evaluate_batch(envsim.bestnns.h, obsbatch)

pbatch_vbatchres = evaluate_batch(envres.bestnns.f, statebatchres)
pbatch_vbatchsim = evaluate_batch(envsim.bestnns.f, statebatchsim)
envsim.bestnns.f

rbatch_snextbatch = evaluate_batch(envres.bestnns.g, collect(zip(statebatchres, actionbatch)))
rbatch_snextbatch = evaluate_batch(envsim.bestnns.g, collect(zip(statebatchsim, actionbatch)))
size(rbatch_snextbatch[1])

envres.bestnns.f(hiddenstate)

r, s =envres.bestnns.g((hiddenstate,3))

res = Vector{Any}(undef,8)
res[1:8] = collect(rbatch_snextbatch)

##
# for batchifierbatchsize in 1:4
batchifierbatchsize = 32
obsbatch = (rand(rand(envsim.memory).states) for _ in 1:batchifierbatchsize)

statebatchres = (rand(Float32,5,4,128) for _ in 1:batchifierbatchsize)
actionbatch = (rand(1:5) for _ in 1:batchifierbatchsize)
gresbatch = zip(statebatchres, actionbatch)

statebatchsim = (rand(Float32, 64) for _ in 1:batchifierbatchsize)
gsimbatch = zip(statebatchsim, actionbatch)


benchmarks = Dict()

#f
benchmarks["f_res_batch_cpu"] = @benchmark evaluate_batch($(envres.bestnns.f),       $collect(statebatchres))
benchmarks["f_res_batch_gpu"] = @benchmark evaluate_batch($(envres.bestnns.f|>gpu),  $collect(statebatchres))

benchmarks["f_sim_batch_cpu"] = @benchmark evaluate_batch($(envsim.bestnns.f),       $collect(statebatchsim))
benchmarks["f_sim_batch_gpu"] = @benchmark evaluate_batch($(envsim.bestnns.f|>gpu),  $collect(statebatchsim))

#g
benchmarks["g_res_batch_cpu"] = @benchmark evaluate_batch($(envres.bestnns.g),       $collect(gresbatch))
benchmarks["g_res_batch_gpu"] = @benchmark evaluate_batch($(envres.bestnns.g|>gpu),  $collect(gresbatch))

benchmarks["g_sim_batch_cpu"] = @benchmark evaluate_batch($(envsim.bestnns.g),       $collect(gsimbatch))
benchmarks["g_sim_batch_gpu"] = @benchmark evaluate_batch($(envsim.bestnns.g|>gpu),  $collect(gsimbatch))

#h
benchmarks["h_res_batch_cpu"] = @benchmark evaluate_batch($(envres.bestnns.h),       $collect(obsbatch))
benchmarks["h_res_batch_gpu"] = @benchmark evaluate_batch($(envres.bestnns.h|>gpu),  $collect(obsbatch))

benchmarks["h_sim_batch_cpu"] = @benchmark evaluate_batch($(envsim.bestnns.h),       $collect(obsbatch))
benchmarks["h_sim_batch_gpu"] = @benchmark evaluate_batch($(envsim.bestnns.h|>gpu),  $collect(obsbatch))

bench_batchsize[batchifierbatchsize] = benchmarks
# end
##
@profview for _ in 1: 1000; evaluate_batch(g_net, collect(gbatch)); end;


# FileIO.save("bench_batchsize2.jld2", "bench_batchsize", bench_batchsize)
bench_batchsize = FileIO.load("bench_batchsize2.jld2", "bench_batchsize")
using BenchmarkPlots, StatsPlots


X = 4:80
# for x in X
  ts = bench_batchsize[x]["h_res_batch_gpu"].times
  t̄ = mean(ts)
  t̃ = median(ts)
  σ = std(ts)



scatter([x],[t̃])

plot!([x],[t̄], ribbon=[σ])
plotly()
bench_batchsize[x]["h_res_batch_gpu"]


evaluate_batch((envres.bestnns.f),       collect(statebatchres))
evaluate_batch((envres.bestnns.f|>gpu),  collect(statebatchres))

evaluate_batch((envsim.bestnns.f),       collect(statebatchsim))
evaluate_batch((envsim.bestnns.f|>gpu),  collect(statebatchsim))

#g
evaluate_batch((envres.bestnns.g),       collect(gresbatch))
typeof(evaluate_batch((player.recurrent_inference),       collect(gresbatch))[1][3])
evaluate_batch((envres.bestnns.g|>gpu),  collect(gresbatch))

evaluate_batch((envsim.bestnns.g),       collect(gsimbatch))
evaluate_batch((envsim.bestnns.g|>gpu),  collect(gsimbatch))

#h
evaluate_batch((envres.bestnns.h),       collect(obsbatch))
evaluate_batch((init),       collect(obsbatch))
evaluate_batch((envres.bestnns.h|>gpu),  collect(obsbatch))

evaluate_batch((envsim.bestnns.h),       collect(obsbatch))
evaluate_batch((envsim.bestnns.h|>gpu),  collect(obsbatch))

evaluate(init,collect(obsbatch)[1])

# state = GI.current_state(game)
# obsbatch = [state, state]

nns = deepcopy(envres.bestnns)|>Flux.gpu|>Flux.testmode!
player = MuPlayer((InitialOracle(nns), RecurrentOracle(nns)), envres.params.self_play.mcts)

collected_obsbatch = collect(obsbatch)
@profview for _ in 1:1000; initres = evaluate_batch((player.initial_inference), collected_obsbatch); end;
@timed for _ in 1:1000; initres = evaluate_batch((player.initial_inference), collected_obsbatch); end;
@profview for _ in 1:1000; recurres = evaluate_batch((player.recurrent_inference), collect(gresbatch)); end;
@timed for _ in 1:1000; recurres = evaluate_batch((player.recurrent_inference), collect(gresbatch)); end;


collect(obsbatch)
typeof(recurres)
typeof(initres)

p = rand(Float32,5,32)
v = rand(Float32,1,32)
s = rand(Float32,5,4,128,32)

cp = cp|>Flux.gpu
cv = cv|>Flux.gpu
cs = cs|>Flux.gpu
@benchmark (p,v,s)|>Flux.gpu
@benchmark (cp,cv,cs)|>Flux.cpu
@benchmark map(Flux.cpu, (cp,cv,cs))
@benchmark map(x->Flux.unstack(x,ndims(x)), (p,v,s))
@benchmark map(x->Flux.unstack(x,ndims(x)), (cp,cv,cs))
@benchmark P,V,S = map($convert_output, $(cp,cv,cs))

@benchmark evaluate_batch($(player.initial_inference), $collect(obsbatch));

P, V, S = map(convert_output, (cp,cv,cs))
@benchmark V = [v[1] for v in $V]
@code_typed evaluate_batch((player.initial_inference), collect(obsbatch))


# only forward
X = Flux.batch(GI.vectorize_state(gspec, b) for b in obsbatch) #obsrvation
Xnet = to_nndevice(player.initial_inference.f, X)
function initforwardonly(init, Xnet)
  S = forward(init.h, Xnet)
  P, V = forward(init.f, S)
  return P, V, S
end
function recurforwardonly(recur, S_A_net)
  R, S = forward(recur.g, S_A_net)
  P, V = forward(recur.f, S)
  return P, V, S, R
end
S = Flux.batch(b[1] for b in gresbatch)
A = Flux.batch(encode_a(gspec, b[2]; batchdim=4) for b in gresbatch)
S_A = cat(S,A,dims=3)
S_A_net = S_A|>gpu

@timed for _ in 1:1000; recurforwardonly(player.recurrent_inference, S_A_net); end;


P,V,S = initforwardonly((player.initial_inference), Xnet)
@benchmark P|>Flux.cpu
@benchmark V|>Flux.cpu
@benchmark Flux.cpu(S)
@benchmark 

@benchmark initforwardonly($(player.initial_inference), Xnet)
@profview for _ in 1:1000; P,V,S = initforwardonly((player.initial_inference), Xnet); end;
@timed for _ in 1:1000; P,V,S = initforwardonly((player.initial_inference), Xnet); end;

function newhash(x)
  x = vec(x)
  h = hash(x)
  # x = reshape(x, 5,4,3)
  return h, x
end

@benchmark hash($rand(Float32,5,4,128))
@benchmark hash($rand(Float32,5,4,3))
@benchmark hash($rand(Float32,64))
@benchmark hash($rand(Float32,32))
@benchmark hash($GI.current_state(game))
@benchmark flatten($rand(Float32,5,4,3))
@benchmark reshape($rand(Float32,60), 5,4,3)
@benchmark newhash($rand(Float32, 5,4,3))

@enter hash(rand(Float32,5,4,128))

@benchmark hash($rand(Int))
@benchmark hash($rand(Int8))
@benchmark hash(($rand(Float32,5,4,3),$rand(Float32)))
@benchmark hash(($rand(Float32,5,4,3), $rand(Int)))
@benchmark hash($rand(Float32,5,12))

@benchmark hash($rand(128,128,128))
@benchmark hash($vec(rand(128,128,128)))
@benchmark hash($rand(128^3))

Base.summarysize(rand(128^3))
Base.summarysize(rand(128,128,128))