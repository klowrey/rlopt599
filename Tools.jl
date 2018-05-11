__precompile__()

module Tools

#################### utilities
function discount_sum{DT<:AbstractFloat}(x::Vector{DT}, gamma::DT, terminal::DT)
   y = similar(x)
   runsum = terminal
   for t=length(x):-1:1
      runsum = x[t] + gamma*runsum
      y[t] = runsum
   end
   return y
end

function discount_sum!{DT<:AbstractFloat}(y::Matrix{DT}, x::Matrix{DT},
                                          t::Int, gamma::DT, terminal::DT)
   runsum = terminal
   for i=size(x, 1):-1:1
      runsum = x[i,t] + gamma*runsum
      y[i,t] = runsum
   end
end

function discount_sum!{T<:AbstractFloat,
                       DT<:AbstractFloat}(y::Matrix{T}, x::Matrix{DT},
                                          t::Int, gamma::DT, terminal::DT)
   runsum = terminal
   for i=size(x, 1):-1:1
      runsum = x[i,t] + gamma*runsum
      y[i,t] = runsum
   end
end

# reward: T x numT
function compute_returns!{DT<:AbstractFloat}(returns::Vector{DT},
                                             rewards::Matrix{Float64},
                                             gamma::Float64)
   T, numT = size(rewards)
   mat_ret = reshape(returns, T, numT)
   for t=1:numT
      discount_sum!(mat_ret, rewards, t, gamma, 0.0)
   end
end

function compute_advantages(returns::Vector{Float64},
                            baseline::Vector{Float64},
                            obs::Matrix{Float64},
                            gamma::Float64)
   return returns - baseline
end

function computeGAEadvantages!{DT<:AbstractFloat}(adv::Vector{DT},
                                                  baseline::Vector,
                                                  rewards::Matrix{Float64},
                                                  obs::Matrix{Float64},
                                                  gae::Float64,
                                                  gamma::Float64)
   T, numT = size(rewards)
   tdsum = Array{DT}(size(rewards))
   mat_adv = reshape(adv, T, numT)

   nthread = min(numT, Threads.nthreads()) # cant have more threads than data
   Threads.@threads for tid=1:nthread # WHY IS MT SO MUCH SLOWER??
      thread_range = Distributed.splitrange(numT, nthread)[tid]

      for t=thread_range
         base = view(baseline, (t-1)*T+1:t*T)
         for k=1:T-1
            tdsum[k,t] = rewards[k,t] + gamma * base[k+1] - base[k]
         end
         tdsum[T,t] = rewards[T,t] - base[T]
         #mat_adv[:,t] = discount_sum(tdsum[:,t], DT(gae*gamma), DT(0.0))
         discount_sum!(mat_adv, tdsum, t, DT(gae*gamma), DT(0.0))
      end
   end
end

# solver algorithms
function cpu_cg_solve{T<:AbstractFloat}(fim::Matrix{T}, # ((n+1)*m + m) X T*numT
                                        vpg::Vector{T}, # ((n+1)*m + m)
                                        cg_iters::Integer=10,
                                        reg::Float64=1e-4,
                                        tol::Float64=1e-10)
   # Initialize cg variables
   r = copy(vpg)
   p = copy(vpg)
   x = zeros(T, size(vpg))  # I want x to be same shape as vpg but full of zeros
   rdr = dot(vpg, vpg)  # outputs a scalar
   z = fim*p
   #z += p*reg 

   iters = 1
   for i=1:cg_iters
      v = rdr/dot(p, z)      # scalar

      x .+= v.*p
      r .-= v.*z

      rdr_new = dot(r, r)    # scalar
      ratio = rdr_new/rdr
      rdr = rdr_new

      iters = i
      if rdr < tol
         break   # this should break the for loop
      end

      p .*= ratio
      p .+= r

      #println(norm(z))
      A_mul_B!(z, fim, p)
   end
   println("    Used $iters cojugate-gradient iterations.");
   return x
end  

function hvp_cg_solve{DT<:AbstractFloat}(hvpfim,
                                         vpg::Vector{DT}, # ((n+1)*m + m)
                                         cg_iters::Integer=10,
                                         reg::Float64=1e-4,
                                         tol::Float64=1e-10)
   # Initialize cg variables
   r = copy(vpg)
   p = copy(vpg)
   x = zeros(DT, size(vpg))
   rdr = dot(vpg, vpg)  # outputs a scalar
   z = zeros(DT, size(vpg))
   hvpfim(z, p)

   iters = 1
   for i=1:cg_iters
      v = rdr/dot(p, z)      # scalar

      x += v*p
      r -= v*z
      #@. x += v*p
      #@. r -= v*z

      rdr_new = dot(r, r)    # scalar
      ratio = rdr_new/rdr
      rdr = rdr_new

      iters = i
      if rdr < tol
         break   # this should break the for loop
      end

      @. p = r + ratio*p
      #p[:] .*= ratio
      #p[:] += r

      #println(norm(z))
      hvpfim(z, p)
   end
   println("    Used $iters HVP-cg iterations.");
   return x
end  

end

