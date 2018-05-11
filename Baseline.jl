
__precompile__()

module Baseline
using Flux
abstract type AbstractBaseline end

struct Linear{DT<:AbstractFloat} <: AbstractBaseline
   nfeat::Int
   T::Int
   numT::Int
   reg_coeff::DT
   coeff::Vector{DT} # states x N

   # workspace
   featnd::Array{DT,3}
   A::Matrix{DT}
   bline::Vector{DT}
   function Linear{DT}(reg::AbstractFloat, ns::Integer, T::Integer, numT::Integer) where DT<:AbstractFloat
      nfeat = 2*ns+3+1
      featnd=zeros(DT, nfeat, T, numT)
      A = zeros(DT, nfeat, nfeat)
      bline = zeros(DT, T*numT)
      return new(nfeat, T, numT, reg, zeros(DT, nfeat),
                 featnd, A, bline)
   end
end

struct Quadratic{DT<:AbstractFloat} <: AbstractBaseline
   nfeat::Int
   T::Int
   numT::Int
   reg_coeff::DT
   coeff::Vector{DT} # states x N

   # workspace
   featnd::Array{DT,3}
   A::Matrix{DT}
   bline::Vector{DT}
   function Quadratic{DT}(reg::AbstractFloat, ns::Integer, T::Integer, numT::Integer) where DT<:AbstractFloat
      nfeat = ns+convert(Int,ns*(ns+1)/2)+ns+ns+4+1
      featnd=zeros(DT, nfeat, T, numT)
      gettimefeatures!(featnd)

      A = zeros(DT, nfeat, nfeat)
      bline = zeros(DT, T*numT)
      return new(nfeat, T, numT, reg, zeros(DT, nfeat),
                 featnd, A, bline)
   end
end

struct NN{DT<:AbstractFloat} <: AbstractBaseline
   nhidden::Int
   nfeat::Int
   T::Int
   numT::Int

   loss::Function
   opt::Function

   #workspace
   featnd::Array{DT,2}
   m::Any
   #w::Array{Any}
   bline::Vector{DT}
   function NN{DT}(ns::Integer, T::Integer, numT::Integer, nhidden::Integer) where DT<:AbstractFloat
      nfeat = 1*ns+3+1
      featnd=zeros(DT, nfeat, T*numT)
      m = Chain(Dense(nfeat, nhidden, tanh),
                Dense(nhidden, nhidden, tanh),
                Dense(nhidden, 1)) |> gpu
      bline = zeros(DT, T*numT)
      return new(nhidden, nfeat, T, numT,
                 (a, b) -> sum((m(a) - b).^2)/length(b),
                 ADAM(params(m)), #, 0.0001),
                 featnd, m, bline)
   end
end


######################################################################## common

function gettimefeatures!{DT<:AbstractFloat}(feat::Array{DT,3})
   T    = size(feat, 2)
   numT = size(feat, 3)
   al = convert(Array{DT}, linspace(1, T, T)') / DT(T)
   for t=1:numT
      feat[end-4,:,t] = al
      feat[end-3,:,t] = al.^2
      feat[end-2,:,t] = al.^3
      feat[end-1,:,t] = al.^4
      feat[end,:,t]   = DT(1.0)
   end
end

function getlinearfeatures(obs::Matrix{Float64})
   T = size(obs,2)
   # TODO add clipping
   o = max.(obs, -10.0)
   o = min.(o, 10.0)
   al = linspace(1, T, T)' / T
   feat = vcat(o, o.^2, al, al.^2, al.^3, ones(1, T))
end

function getquadfeatures!{DT<:AbstractFloat}(feat::Array{DT,3},
                                             obs::Matrix{DT},
                                             t::Integer)
   n = size(obs, 1)
   T = size(feat, 2)

   feat[1:n,:,t] = obs[:,(t-1)*T+1:t*T] #implicite conversion between 32, 64
   feat[1:n,:,t] = max.(feat[1:n,:,t], DT(-10.0))
   feat[1:n,:,t] = min.(feat[1:n,:,t], DT(10.0))
   feat[1:n,:,t] /= DT(10.0)

   qfsize = convert(Int, n*(n+1)/2)
   qf = view(feat, (n+1):(n+qfsize) , :, t) #Matrix{Float64}(qfsize, T)
   k = 1
   @inbounds for i=1:n
      for j=i:n
         for x=1:T
            qf[k,x] = feat[i,x,t].*feat[j,x,t]
         end
         k += 1
      end
   end

   i0 = n+qfsize
   i1 = i0+n
   @inbounds for j = 1:n
      for i=1:T
         feat[i0+j,i,t] = sin(pi*feat[j,i,t])
         feat[i1+j,i,t] = cos(pi*feat[j,i,t])
      end
   end
end

function getNNfeatures(obs::Matrix{Float64})
   T = size(obs,2)
   # TODO add clipping
   o = max.(obs, -10.0)
   o = min.(o, 10.0)
   # al = linspace(1, T, T)' / 100.0
   al = linspace(1, T, T)' / 1000.0
   # feat = vcat(o, o.^2, al, al.^2, al.^3, ones(1, T))
   feat = vcat(o, al, al.^2, al.^3, ones(1, T))
end

function prefit!{DT<:AbstractFloat}(b::AbstractBaseline, returns::Vector{DT})
   featmat = reshape(b.featnd, b.nfeat, b.numT*b.T) 
   #b.A[:] = (featmat*featmat')
   A_mul_Bt!(b.A, featmat, featmat)

   return featmat*returns
end

# returns: T*numT
function fit!{DT<:AbstractFloat}(b::AbstractBaseline, returns::Vector{DT})

   target = prefit!(b, returns) # sets A matrix, returns target

   for i=1:b.nfeat # set diag
      b.A[i,i] += b.reg_coeff
   end
   #b.coeff[:] = b.A\(target)
   if isposdef(b.A)
      A_ldiv_B!(b.coeff, cholfact!(b.A), target) # needs A to be factorized
   else
      A_ldiv_B!(b.coeff, lufact!(convert(Array{Float64}, b.A)), target) # needs A to be factorized
   end
end

function fit!{DT<:AbstractFloat}(b::NN, returns::Vector{DT})
   nsamples = length(returns)
   const nbatch = 50

   X = b.featnd |> gpu
   Y = returns |> gpu

   rndidx = Flux.chunk(shuffle(1:nsamples), nbatch)

   datas = [ (X[:,i], Y[i]') for i in rndidx ]
   
   opt = ADAM(params(b.m))
   evalcb() = @show(b.loss(b.featnd, returns'))

   BLAS.set_num_threads(Threads.nthreads())
   for i=1:3
      Flux.train!(b.loss, datas, b.opt)#, cb = Flux.throttle(evalcb, 1))
   end

end

######################################################################## linear

function predict!{T<:AbstractFloat}(b::Linear, obs::Matrix{T})
   nthread = min.(b.numT, Threads.nthreads()) # cant have more threads than data
   Threads.@threads for tid=1:nthread
      #for tid=1:nthread
      thread_range = Distributed.splitrange(b.numT, nthread)[tid]

      for t=thread_range
         b.featnd[:,:,t] = getlinearfeatures(obs[:,(t-1)*b.T+1:t*b.T])
      end
   end
   b.bline[:] = reshape(b.featnd, b.nfeat, b.numT*b.T)'*b.coeff
end

##################################################################### quadratic
# TODO this is hacky and this whole module is a disgrace
function predict{DT<:AbstractFloat}(b::Quadratic, obs::Matrix{DT})
   nsamples = size(obs, 2)
   featnd=zeros(DT, b.nfeat, 1, nsamples)
   gettimefeatures!(featnd)
   for t=1:nsamples
      getquadfeatures!(featnd, obs, t)
   end
   values = reshape(featnd, b.nfeat, nsamples)'*b.coeff
   return values
end

function predict!{DT<:AbstractFloat}(b::Quadratic, obs::Matrix{DT})
   nthread = min.(b.numT, Threads.nthreads()) # cant have more threads than data
   Threads.@threads for tid=1:nthread
      #for tid=1:nthread
      thread_range = Distributed.splitrange(b.numT, nthread)[tid]

      for t=thread_range
         getquadfeatures!(b.featnd, obs, t)
      end
   end
   b.bline[:] = reshape(b.featnd, b.nfeat, b.numT*b.T)'*b.coeff
end

##################################################################### Neural Network

function predict!{T<:AbstractFloat}(b::NN, obs::Matrix{T})
   b.featnd[:,:] = getNNfeatures(obs)
   b.bline .= b.m(b.featnd).data[1,:]
end

end # module
