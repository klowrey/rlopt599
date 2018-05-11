
__precompile__()

module PolicyGradient

using mjWrap
using LearningStrategies
import LearningStrategies: setup!, update!
using Baseline
using Policy
using Tools
using ExpFunctions

using ReverseDiff

using IterativeSolvers

export PolicyGradModel, NPGStrategy, setup!, update!, PolicyRollout

struct PolicyGradModel
   theta::Vector     # policy parameters
   mjsys::mjWrap.mjSet
   policy::Policy.AbstractPolicy

   # scratch stuff
   samples::mjWrap.TrajSamples
   meansamples::mjWrap.TrajSamples

   trace::Dict{Symbol,Vector{Float64}}

   niter::Int

   function PolicyGradModel(theta,
                            mjsys, policy,
                            T, numT, meanT, niter)
      ns = mjsys.ns
      samples     = mjw.allocateTrajSamples(mjsys, T, numT, ns)
      meansamples = mjw.allocateTrajSamples(mjsys, T, meanT, ns)
      return new(theta, mjsys, policy,
                 samples, meansamples,
                 Dict(:stocR => Vector{Float64}(),
                      :meanR => Vector{Float64}(),
                      :evalscore => Vector{Float64}() ), niter)
                 #zeros(niter), zeros(niter), zeros(niter), niter)
   end
end

struct NPGStrategy{dtype<:AbstractFloat} <: LearningStrategy
   vpg::Vector{dtype}
   npg::Vector{dtype}
   grad::Vector{dtype}

   fullFIM::Bool
   gradll::Matrix{dtype}
   FIM::Matrix{dtype}

   adv::Vector{dtype}
   ret::Vector{dtype}

   baseline::Baseline.AbstractBaseline

   norm_step_size::Float64
   gamma::Float64
   gae::Float64

   cg_iter::Int
   cg_reg::Float64
   cg_tol::Float64

   obsmat::Matrix{dtype}
   ctrlmat::Matrix{dtype}

   function NPGStrategy{dtype}(model,
                               baseline,
                               fullFIM,
                               norm_step_size,
                               gamma, gae,
                               cg_iter, cg_reg, cg_tol) where dtype<:AbstractFloat
      #dtype = eltype(model.policy.theta) #Float64
      T, numT = model.samples.T, model.samples.numT

      nparam  = length(model.theta)
      vpg     = zeros(dtype, nparam)
      npg     = zeros(dtype, nparam)
      grad    = zeros(dtype, nparam)

      if fullFIM
         gradll  = zeros(dtype, nparam, T*numT)
         FIM     = zeros(dtype, nparam, nparam)
      else
         gradll  = zeros(dtype, 1, 1)
         FIM     = zeros(dtype, 1, 1)
      end

      adv     = zeros(dtype, T*numT)
      ret     = zeros(dtype, T*numT)

      tmpobs  = rand(dtype, model.mjsys.ns, T*numT)
      tmpctrl = rand(dtype, model.mjsys.nu, T*numT)

      new(vpg, npg, grad,
          fullFIM, gradll, FIM, adv, ret,
          baseline,
          norm_step_size,
          gamma, gae,
          cg_iter, cg_reg, cg_tol,
          tmpobs, tmpctrl)
   end
end
Base.show(io::IO, s::NPGStrategy) = print(io, "NPG: $(s.fullFIM?"FIM":"HVP"), δ=$(s.norm_step_size)\n")


function grad_builder(f::Function, input)
   ctp = ReverseDiff.GradientTape(f, input)
   if length(ctp.tape) <= 10000
      info("Compiling tape")
      ctp = ReverseDiff.compile(ctp)
   end
   return (out,in)->ReverseDiff.gradient!(out, ctp, in) # taped 
end

# feature of learning strats; need to have iter and null to get iteration num
function update!(model::PolicyGradModel, s::NPGStrategy, iter, null)

   tic()

   samples  = model.samples
   mjsys    = model.mjsys
   pol      = model.policy

   b        = s.baseline

   nsamples = samples.T*samples.numT

   # Parallize here......................................................
   rsum = sum(samples.reward, 1) # rollouts done in different strat

   if eltype(pol.theta) == Float64
      ctrlmat = reshape(samples.ctrl, mjsys.nu, nsamples)
      obsmat  = reshape(samples.obs, mjsys.ns, nsamples) 
   else
      copy!(s.ctrlmat, convert(Matrix{Float32}, reshape(samples.ctrl, mjsys.nu, nsamples)))
      copy!(s.obsmat, convert(Matrix{Float32}, reshape(samples.obs, mjsys.ns, nsamples)))
      ctrlmat = s.ctrlmat
      obsmat  = s.obsmat
   end

   println("returns")
   @time Tools.compute_returns!(s.ret, samples.reward, s.gamma)

   println("predict")
   @time Baseline.predict!(b, reshape(samples.obs, mjsys.ns, nsamples))

   println("advantages")
   @time Tools.computeGAEadvantages!(s.adv, b.bline,
                                     samples.reward,
                                     reshape(samples.obs, mjsys.ns, nsamples), s.gae, s.gamma)

   println("fit")
   @time Baseline.fit!(b, s.ret) # update fit

   if s.fullFIM
      DT = eltype(s.vpg)
      
      info("FULL FIM")
      println("gradll")
      @time Policy.gradLL!(s.gradll, pol, obsmat, ctrlmat)

      println("vpg")
      # vpg[:] = gradll*adv / nsamples
      @time BLAS.gemv!('N', 1/DT(nsamples), s.gradll, s.adv, DT(0.0), s.vpg)

      println("fim")
      #FIM[:] = gradll*gradll' / nsamples
      @time begin
         #const α = 0.9
         #FIM = copy(s.FIM)
         A_mul_Bt!(s.FIM, s.gradll, s.gradll)
         scale!(s.FIM, 1/DT(nsamples))
         #@. s.FIM = FIM + α*(s.FIM - FIM) # keep older FIM data around; works for higher α
      end

      println("cg")
      @time s.npg[:] = Tools.cpu_cg_solve(s.FIM, s.vpg, s.cg_iter, s.cg_reg, s.cg_tol)
      #@time IterativeSolvers.cg!(s.npg, s.FIM, s.vpg; tol = s.cg_tol, maxiter = s.cg_iter)
   else
      info("HVP algorithm")
      @time my∇LLxA = grad_builder( (theta, a)->Policy.loglikelihood(theta, pol, obsmat, ctrlmat)*a,
                                   (pol.theta, s.adv) )
      println("vpg")
      @time begin
         my∇LLxA((s.vpg, s.adv), (pol.theta, s.adv))
         scale!(s.vpg, 1/nsamples)
      end

      # TODO don't need in-place Loglikelihood for output; need inplace during comp
      minus = similar(s.adv')
      alpha = similar(s.adv')

      function hvp!(z, vec)
         const ϵ = sqrt(eps(eltype(vec))) * norm(vec)
         const eps_vec = ϵ * vec
         Policy.loglikelihood!(alpha, pol.theta+eps_vec, pol, obsmat, ctrlmat)
         Policy.loglikelihood!(minus, pol.theta-eps_vec, pol, obsmat, ctrlmat)
         alpha .-= minus
         scale!(alpha, 1.0/2ϵ)
         s.adv .= alpha'
         my∇LLxA((z, s.adv), (pol.theta, s.adv))
         #scale!(z, 1/nsamples)
      end
      println("cg")
      @time s.npg[:] = Tools.hvp_cg_solve(hvp!, s.vpg, s.cg_iter, s.cg_reg, s.cg_tol)
      scale!(s.npg, nsamples)
   end

   alpha = sqrt(s.norm_step_size / dot(s.npg, s.vpg))
   println("grad")
   s.grad .= alpha.*s.npg

   println("update")
   Policy.updatetheta(pol, pol.theta + s.grad) # update parameters
   model.theta[:] = pol.theta

   ########################################################## mean rollout 
   println("no noise")
   meanreward = sum(model.meansamples.reward, 1)

   ############################################################### cleanup
   iterscore = mean(rsum)
   meanscore = mean(meanreward)
   if iter > 1
      @printf "\n  Stoc: %f, Mean: %f" iterscore meanscore
   else
      @printf "\n  Mean of rewards: %f" iterscore
   end
   @printf "\n         NPG Norm: %f\n\n" norm(s.npg) #normgrad[iter] 
   @printf "  Reward of Mean Rollout: %f\n" meanscore
   itertime = toc()
   @printf "  Total time: %f\n" itertime
   push!(model.trace[:stocR], iterscore)
   push!(model.trace[:meanR], meanscore)
end

mutable struct PolicyRollout <: LearningStrategy
   f::FunctionSet
   policyaction::Function
   policymean::Function
   function PolicyRollout(f)
      new(f, ()->nothing, ()->nothing)
   end
end
Base.show(io::IO, s::PolicyRollout) = print(io, "π rollout\n")

function setup!(s::PolicyRollout, model::PolicyGradModel)
   s.policyaction = (c,o)->Policy.getaction!(c, model.policy, o)
   s.policymean   = (c,o)->Policy.getmean!(c, model.policy, o)
end

function stochasticrollout(mjsys::mjWrap.mjSet,
                           samples::mjWrap.TrajSamples,
                           ctrlfunc!::Function, f::FunctionSet, iter::Int)
   # stochastic rollout
   f.setmodel!(mjsys, iter) # change model for this iteration
   f.initstate!(mjsys, samples)
   f.setcontrol!(mjsys, samples.ctrl) # sets policy noise in ctrl
   @time mjWrap.rollout(mjsys, samples,
                        ctrlfunc!, f.observe!, f.reward)
end

function update!(model::PolicyGradModel, s::PolicyRollout, iter, null)
   println("policy rollout, stoc & mean")

   mjsys       = model.mjsys
   samples     = model.samples
   meansamples = model.meansamples

   tic()
   stochasticrollout(mjsys, samples, s.policyaction, s.f, iter)

   # policy's mean rollout
   s.f.initstate!(mjsys, meansamples)
   mjWrap.rollout(mjsys, meansamples,
                  s.policymean,
                  s.f.observe!, s.f.reward)
   toc()
end

end
