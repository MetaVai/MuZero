# """
#    fill_and_evaluate(net, batch; batch_size, fill_batches)

# Evaluate a neural network on a batch of inputs.

# If `fill_batches=true`, the batch is padded with dummy inputs until
# it has size `batch_size` before it is sent to the network.

# This function is typically called by inference servers.
# """
function fill_and_evaluate(net, batch; batch_size, fill_batches)
  n = length(batch)
  @assert n > 0
  if !fill_batches
    return evaluate_batch(net, batch)
  else
    nmissing = batch_size - n
    @assert nmissing >= 0
    if nmissing == 0
      return evaluate_batch(net, batch)
    else
      batch = vcat(batch, [batch[1] for _ in 1:nmissing])
      return evaluate_batch(net, batch)[1:n]
    end
  end
end

# function launch_inference_server(
#   net1::MuNetwork,
#   net2::MuNetwork;
#   num_workers,
#   batch_size,
#   fill_batches)

#   return Batchifier.launch_server(;num_workers, batch_size) do batch
#     n = length(batch) # batch is vector Any, cause h(O), f(S), and g(S,A) are evaluated

#     maskf1 = findall(b->b.netid == :f1, batch)
#     maskg1 = findall(b->b.netid == :g1, batch)
#     maskh1 = findall(b->b.netid == :h1, batch)

#     maskf2 = findall(b->b.netid == :f2, batch)
#     maskg2 = findall(b->b.netid == :g2, batch)
#     maskh2 = findall(b->b.netid == :h2, batch)
#     @assert length(maskf1) + length(maskg1) + length(maskh1) +
#             length(maskf2) + length(maskg2) + length(maskh2) == n

#     batchf1 = [b.query for b in batch[maskf1]]
#     batchg1 = [b.query for b in batch[maskg1]]
#     batchh1 = [b.query for b in batch[maskh1]]

#     batchf2 = [b.query for b in batch[maskf2]]
#     batchg2 = [b.query for b in batch[maskg2]]
#     batchh2 = [b.query for b in batch[maskh2]]

#     resf1 = isempty(maskf1) ? Nothing[] : fill_and_evaluate(net1.f, batchf1; batch_size=n, fill_batches)
#     resg1 = isempty(maskg1) ? Nothing[] : fill_and_evaluate(net1.g, batchg1; batch_size=n, fill_batches)
#     resh1 = isempty(maskh1) ? Nothing[] : fill_and_evaluate(net1.h, batchh1; batch_size=n, fill_batches)

#     resf2 = isempty(maskf2) ? Nothing[] : fill_and_evaluate(net2.f, batchf2; batch_size=n, fill_batches)
#     resg2 = isempty(maskg2) ? Nothing[] : fill_and_evaluate(net2.g, batchg2; batch_size=n, fill_batches)
#     resh2 = isempty(maskh2) ? Nothing[] : fill_and_evaluate(net2.h, batchh2; batch_size=n, fill_batches)


#     res = Vector{Union{eltype.((resf1,resg1,resh1,resf2,resg2,resh2,))...}}(undef, n)

#     res[maskf1] = resf1
#     res[maskg1] = resg1
#     res[maskh1] = resh1

#     res[maskf2] = resf2
#     res[maskg2] = resg2
#     res[maskh2] = resh2
#     return res
#   end
# end


# function launch_inference_server(
#   net::MuNetwork;
#   num_workers,
#   batch_size,
#   fill_batches)

#   return Batchifier.launch_server(;num_workers, batch_size) do batch
#     n = length(batch) # batch is vector Any, cause h(O), f(S), and g(S,A) are evaluated
#     maskf = findall(b->b.netid == :f, batch)
#     maskg = findall(b->b.netid == :g, batch)
#     maskh = findall(b->b.netid == :h, batch)
#     @assert length(maskf) + length(maskg) + length(maskh) == n

#     batchf = [b.query for b in batch[maskf]]
#     batchg = [b.query for b in batch[maskg]]
#     batchh = [b.query for b in batch[maskh]]

#     resf = isempty(maskf) ? Nothing[] : fill_and_evaluate(net.f, batchf; batch_size=n, fill_batches)
#     resg = isempty(maskg) ? Nothing[] : fill_and_evaluate(net.g, batchg; batch_size=n, fill_batches)
#     resh = isempty(maskh) ? Nothing[] : fill_and_evaluate(net.h, batchh; batch_size=n, fill_batches)

#     res = Vector{Union{eltype.((resf,resg,resh))...}}(undef, n)
#     res[maskf] = resf
#     res[maskg] = resg
#     res[maskh] = resh
#     return res
#   end
# end

function launch_inference_server(
  net::Tuple{InitialOracle, RecurrentOracle};
  num_workers,
  batch_size,
  fill_batches)

  return Batchifier.launch_server(;num_workers, batch_size) do batch
    n = length(batch) # batch is vector Any, cause h(O), f(S), and g(S,A) are evaluated
    mask_i = findall(b->b.netid == :i, batch)
    mask_r = findall(b->b.netid == :r, batch)
    @assert length(mask_i) + length(mask_r) == n

    batch_i = [b.query for b in batch[mask_i]]
    batch_r = [b.query for b in batch[mask_r]]

    if isempty(mask_i)
      return fill_and_evaluate(net[2], batch_r; batch_size=n, fill_batches)
    elseif isempty(mask_r)
      return fill_and_evaluate(net[1], batch_i; batch_size=n, fill_batches)
    else
      res_i = fill_and_evaluate(net[1], batch_i; batch_size=n, fill_batches)
      res_r = fill_and_evaluate(net[2], batch_r; batch_size=n, fill_batches)
      @assert typeof(res_i) == typeof(res_r)
      res = Vector{eltype(res_i)}(undef, n)
      res[mask_i] = res_i
      res[mask_r] = res_r
      return res
    end
  end
end

function launch_inference_server(
  net1::Tuple{InitialOracle, RecurrentOracle},
  net2::Tuple{InitialOracle, RecurrentOracle};
  num_workers,
  batch_size,
  fill_batches)

  return Batchifier.launch_server(;num_workers, batch_size) do batch
    n = length(batch) # batch is vector Any, cause h(O), f(S), and g(S,A) are evaluated
    mask_i  = findall(b->b.netid == :i1, batch)
    mask_r  = findall(b->b.netid == :r1, batch)
    mask_i2 = findall(b->b.netid == :i2, batch)
    mask_r2 = findall(b->b.netid == :r2, batch)
    @assert length(mask_i) + length(mask_r) + length(mask_i2) + length(mask_r2) == n

    batch_i = [b.query for b in batch[mask_i]]
    batch_r = [b.query for b in batch[mask_r]]
    batch_i2 = [b.query for b in batch[mask_i2]]
    batch_r2 = [b.query for b in batch[mask_r2]]


    res_r = isempty(mask_r) ? Nothing[] : fill_and_evaluate(net1[2], batch_r; batch_size=n, fill_batches)
    res_r2 = isempty(mask_r2) ? Nothing[] : fill_and_evaluate(net2[2], batch_r2; batch_size=n, fill_batches)
    res_i = isempty(mask_i) ? Nothing[] : fill_and_evaluate(net1[1], batch_i; batch_size=n, fill_batches)
    res_i2 = isempty(mask_i2) ? Nothing[] : fill_and_evaluate(net2[1], batch_i2; batch_size=n, fill_batches)
    
    # @assert typeof(res_i) == typeof(res_r) == typeof(res_i2) == typeof(res_r2)
    res = Vector{eltype(res_i)}(undef, n)
    res[mask_i] = res_i
    res[mask_r] = res_r
    res[mask_i2] = res_i2
    res[mask_r2] = res_r2

    return res
  end
end

ret_oracle(x) = () -> x
do_nothing!() = nothing
send_done!(reqcs...) = () -> foreach(Batchifier.client_done!, reqcs)
zipthunk(f1, f2) = () -> (f1(), f2())

function batchify_oracles(o::MuNetwork; kwargs...)
  reqc = launch_inference_server(o; kwargs...)
  make() = (;
  g = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:g)),
  f = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:f)),
  h = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:h)),
  )
  return make, send_done!(reqc)
end

function batchify_oracles(os::Tuple{MuNetwork,MuNetwork}; kwargs...)
  reqc = launch_inference_server(os[1],os[2]; kwargs...)
  make1() = (;
  g = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:g1)),
  f = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:f1)),
  h = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:h1)),
  )
  make2() = (;
  g = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:g2)),
  f = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:f2)),
  h = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:h2)),
  )
  return zipthunk(make1,make2), send_done!(reqc)
end

function batchify_oracles(os::Tuple{<:Any, AbstractNetwork}; kwargs...)
  reqc = launch_inference_server(os[2]; kwargs...)
  make2() = (;
  g = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:g)),
  f = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:f)),
  h = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:h)),
  )
  return zipthunk(ret_oracle(os[1]), make2), send_done!(reqc)
end

function batchify_oracles(os::Tuple{AbstractNetwork, <:Any}; kwargs...)
  reqc = launch_inference_server(os[1]; kwargs...)
  make1() = (;
  g = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:g)),
  f = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:f)),
  h = Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:h)),
  )
  return zipthunk(make1, ret_oracle(os[2])), send_done!(reqc)
end

function batchify_oracles(os::Tuple{<:Any, <:Any}; kwargs...)
  return zipthunk(ret_oracle(os[1]), ret_oracle(os[2])), do_nothing!
end

function batchify_oracles(o::Any; kwargs...)
  return ret_oracle(o), do_nothing!
end

function batchify_oracles(o::Tuple{InitialOracle,RecurrentOracle}; kwargs...)
  reqc = launch_inference_server(o; kwargs...)
  make() = (
    Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:i)),
    Batchifier.BatchedOracle(reqc, q -> (query=q, netid=:r))
  )
  return make, send_done!(reqc)
end