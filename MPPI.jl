
__precompile__()

module MPPI 

using mjWrap
using LearningStrategies
import LearningStrategies: setup!, update!, hook
using ExpFunctions
using Common

export MPPIModel, MPPIStrategy, setup!, update!, hook
export MPPIRollout
export MPCRollout

struct MPPIModel
   theta::Matrix{Float64} # parameters to improve
   mjsys::mjWrap.mjSet    # environment
   s0::Vector{Float64}    # initial / current state
   terminalvalue::Function 

   Σ::Matrix{Float64}
   λ::Float64

   MPC::Bool

   # scratch stuff
   samples::TrajSamples
   traj::TrajSamples

   trace::Dict{Symbol,Vector{Float64}}

   niter::Int

   function MPPIModel(sigma, lambda, mjsys, mpcT, T, numT;
                      theta=zeros(mjsys.nu, mpcT),
                      s0=zeros(mjsys.nq + mjsys.nv)*NaN, # set NaN init to signal Rollout functions
                      valuefunction=(x)->zeros(numT),
                      niter=T)
      ns = mjsys.ns

      if mpcT == T # not MPC mode
         samples = mjw.allocateTrajSamples(mjsys, T, numT, ns)
         traj    = mjw.allocateTrajSamples(mjsys, T, 1, ns)
         mpc     = false
         warn("MPPI in FULL TRAJECTORY Mode")
      else
         samples = mjw.allocateTrajSamples(mjsys, mpcT, numT, ns)
         traj    = mjw.allocateTrajSamples(mjsys, T, 1, ns)
         mpc     = true
         warn("MPPI in Model Predictive Mode")
      end

      return new(theta, mjsys, s0, valuefunction, sigma, lambda, mpc,
                 samples, traj,
                 Dict(:stocR => Vector{Float64}(),
                      :meanR => Vector{Float64}(),
                      :evalscore => Vector{Float64}() ), niter)
   end
end

struct MPPIStrategy <: LearningStrategy
   costs::Vector{Float64}
   weights::Vector{Float64}

   function MPPIStrategy(model)
      dtype   = Float64
      numT    = model.samples.numT

      costs   = Vector{dtype}(numT)
      weights = Vector{dtype}(numT)

      new(costs, weights)
   end
end
Base.show(io::IO, s::MPPIStrategy) = print(io, "MPPI")

function apply(f::Function, costs::Vector, states::Array{Float64,3})
   for k=1:size(states,3)
      costs[k] -= f(@view states[:,end,k])
   end
   #costs .-= f(states[:,end,:]) # block calculate
end

function update!(model::MPPIModel, s::MPPIStrategy, iter, null)

   BLAS.set_num_threads(Threads.nthreads())

   mjsys   = model.mjsys
   samples = model.samples
   T       = samples.T
   numT    = samples.numT

   s.costs[:] = -1.0.*sum(samples.reward, 1) # rollouts done in different strat

   baseline = minimum(s.costs)

   push!(model.trace[:stocR], mean(s.costs))
   push!(model.trace[:meanR], baseline)

   @. s.costs = exp((-(s.costs-baseline)/model.λ))

   η = sum(s.costs)
   s.weights .= s.costs/η

   #SIGMA = true
   #if SIGMA
   #   #println("before:\n",model.Σ)
   #   sig = reshape(reshape(broadcast(-, samples.ctrl, model.theta),
   #                         mjsys.nu*T, numT)*s.weights,
   #                 mjsys.nu, T)
   #   model.Σ .= 0.0
   #   for t=1:T
   #      BLAS.gemm!('N', 'T', 1.0, sig[:,t], sig[:,t], 1.0, model.Σ)
   #   end
   #   model.Σ ./= T
   #   for u=1:mjsys.nu
   #      model.Σ[u,u] .= max(1e-1, model.Σ[u,u])
   #   end
   #   if mod(iter, 100) == 0
   #      println("Σ:\n",mean(model.Σ))
   #   end
   #end

   model.theta .= reshape(reshape(samples.ctrl,
                                  mjsys.nu*T, numT)*s.weights,
                          mjsys.nu, T)

end

############################################################### ROLLOUTS

struct MPPIRollout <: LearningStrategy
   f::FunctionSet
   c::Vector{Float64}
   o::Vector{Float64}
   function MPPIRollout(f)
      new(f, Array{Float64,1}(), Array{Float64,1}())
   end
end
Base.show(io::IO, s::MPPIRollout) = print(io, "MPPI rollout")

function setup!(s::MPPIRollout, model::MPPIModel)
   samples = model.samples

   push!(s.c, zeros(model.mjsys.nu)...)
   push!(s.o, zeros(model.mjsys.ns)...)

   # traj and samples both start from same place; send inital state to MPPIModel
   if isnan(model.s0[1])
      info("Setting initial state from experiment.")
      s.f.initstate!(model.mjsys, model.traj)
      model.s0 .= model.traj.state[:,1,1]
   else
      model.traj.state[:,1,1] .= model.s0
   end
   samples.state[:,1,:] .= model.s0
end

function update!(model::MPPIModel, s::MPPIRollout, iter, null)
   samples = model.samples
   T       = samples.T
   numT    = samples.numT
   mjsys   = model.mjsys

   noise = reshape( model.Σ * randn(mjsys.nu, T*numT), mjsys.nu, T, numT)
   for k=1:numT
      samples.ctrl[:,1,k] = noise[:,1,k]
      for i=2:T-1
         @. samples.ctrl[:,i,k] = 0.25*noise[:,i-1,k]+0.5*noise[:,i,k]+0.25*noise[:,i+1,k]
      end
      samples.ctrl[:,T,k] = noise[:,T,k]
   end

   samples.ctrl[:,:,1] .= model.theta

   broadcast!(+, samples.ctrl, samples.ctrl, model.theta)
   samples.ctrl[:,:,1] .= model.theta # have at least one thing to do a full theta rollout

   #for u=1:mjsys.nu
   #   samples.ctrl[u,:,:] .= clamp.(samples.ctrl[u,:,:],
   #                                 mjsys.m.actuator_ctrlrange[1,u],
   #                                 mjsys.m.actuator_ctrlrange[2,u])
   #end

   mjWrap.rollout(mjsys, samples, (x...)->nothing, s.f.observe!, s.f.reward)
end

function hook(s::MPPIRollout, model::MPPIModel, iter) # just rollout and store in traj
   if model.MPC # get first controls, mj.step, store in model.traj
      mjsys   = model.mjsys
      traj    = model.traj
      samples = model.samples

      # MPC: apply first controls to system; shift the rist around
      s.c .= model.theta[:,1]
      model.theta[:,1:end-1] .= model.theta[:,2:end]
      model.theta[:,end] .= 0.0

      mjWrap.reset(mjsys, 1, model.s0, s.c, s.o)

      mjWrap.step!(mjsys, 1, s.c, model.s0, s.o) # get next state, next observation
      s.f.observe!(mjsys, 1, model.s0, s.o) # arbitrary observation vector manipulation

      # advance our main trajectory
      traj.state[:,iter,1] .= model.s0
      traj.ctrl[:,iter,1]  .= s.c
      traj.obs[:,iter,1]   .= s.o

      # update the mpc bundle of trajectories to start 'here'
      samples.state[:,1,:] .= model.s0
   elseif mod1(iter, model.niter) == model.niter
      #info("Running rollout for main trajectory.")
      mjsys   = model.mjsys
      traj    = model.traj

      traj.ctrl[:,:,1] = model.theta
      mjWrap.rollout(mjsys, traj, (x...)->nothing, s.f.observe!, s.f.reward)
   end
end

end

