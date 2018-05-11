__precompile__()

module Policy

# POLICY module; provides generic functions like gradloglikelihood, etc
# with Gaussian Linear Policy 
# and Gaussian Radial Basis Function Policy
# types

using JLD
using ReverseDiff
using Distributions

abstract type AbstractPolicy end

# TODO may be able to do a getindex trick to address policy layers
struct GLP{T<:AbstractFloat} <: AbstractPolicy
   n::Integer
   m::Integer
   nparam::Integer

   theta::Vector{T}

   ranges::Dict{Symbol,Range}

   function GLP{T}(obsspace::Integer, actspace::Integer) where T<:AbstractFloat
      n = obsspace
      m = actspace
      d = m*n + m + m

      r=Dict{Symbol,Range}()
      r[:W] = (1:m*n)
      r[:b] = (1:m) + r[:W].stop
      r[:ls]= (1:m) + r[:b].stop
      @assert r[:ls].stop == d

      return new(n,m,d,zeros(T, d),r)
   end
end

mutable struct NN{T<:AbstractFloat} <: AbstractPolicy
   n::Integer
   m::Integer
   nparam::Integer
   nhidden::Integer
   nlayers::Integer

   theta::Vector{T}

   ranges::Dict{Symbol,Range}

   scratch1::Matrix{T}
   scratch2::Matrix{T}
   scratch3::Matrix{T}

   function NN{T}(obsspace::Integer, actspace::Integer,
                  nhidden::Integer, nlayers::Integer) where T<:AbstractFloat
      @assert nlayers == 2
      n = obsspace
      m = actspace
      nh = nhidden
      nlay = nlayers - 1 # skim off top layer
      d = (nh*n + nh) +
      (nlay*nh*nh) + nlay*nh +
      m*nh + m +
      m

      r = Dict{Symbol,Range}()
      r[:Win] = (1:n*nh)
      r[:bin] = (1:nh) + r[:Win].stop

      r[:Wh]  = (1:nh*nh*nlay) + r[:bin].stop
      r[:Wh1] = (1:nh*nh)      + r[:bin].stop
      #r[:Wh2] = (1:nh*nh)         + r[:Wh1].stop

      r[:bh]  = (1:nh*nlay) + r[:Wh].stop
      r[:bh1] = (1:nh)      + r[:Wh1].stop
      #r[:bh2] = (1:nh)         + r[:bh1].stop

      #r[:Wout]= (1:nh*m) + r[:bh2].stop
      r[:Wout]= (1:nh*m) + r[:bh1].stop
      r[:bout]= (1:m)    + r[:Wout].stop

      r[:ls]  = (1:m)    + r[:bout].stop
      @assert r[:ls].stop == d

      nn = new(n, m, d, nhidden, nlay, zeros(T, d), r,
               zeros(T, 1,1), zeros(T, 1,1), zeros(T, 1,1))

      Win, _, Wh, _, Wout, _, _ = split_theta!(nn)
      Win[:]  = sqrt(T(1.0)/(n)) * randn(T,nh*n)
      Wh[:]   = sqrt(T(1.0)/nh) * randn(T,nlay*nh*nh)
      Wout[:] = sqrt(T(1e-3)/(nh)) * randn(T,m*nh)

      return nn
   end
end


################################################################# Create Policy

function save_pol(p::AbstractPolicy, skip::Int, modelname::String, filename::String)
   jldopen(filename, "w") do file
      write(file, "policy", p)
      write(file, "skip", skip)
      write(file, "model", modelname)
   end
end

function loadpolicy(filename::String)
   if isdir(filename)
      filename *= "/policy.jld"
   end
   d = load(filename)
   return d["policy"], d["skip"], d["model"]
end

############################################################## Policy Functions

PolType = Union{Array{Float32}, Array{Float64}, ReverseDiff.TrackedArray}

function getls(p::AbstractPolicy)
   ls = reshape( view(p.theta, p.ranges[:ls]), p.m)
   return ls
end

#function loglikelihood{T<:PolType}(theta::T, p::AbstractPolicy,
#                                   features::T, actions::T)
#   N = size(features, 2) # N = T*numT
#   ll = Array{eltype(theta)}(1,N)
#   loglikelihood!(ll, theta, p, features, actions)
#   return ll
#end

function updatetheta{T<:AbstractFloat}(p::AbstractPolicy, newtheta::Vector{T})
   p.theta[:] = newtheta
end

# API:
# each policy needs the following functions:
# getmean!     : mean policy output c = Policy(input)
# getaction!   : policy output with noise stored in c
#              : c = Policy(input) + c * exp(logstd)
# gradLL!      : gradient of loglikelihood function wrt to policy params
# split_theta! : policy is stored as a vector; splits vector into usable chunks
# updatetheta  : updates vector of parameters (if different than copy)

include("./GLPolicy.jl")
include("./NNPolicy.jl")

end # module Policy
