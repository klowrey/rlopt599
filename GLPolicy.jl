########################################################################### GLP

function getmean!{T<:AbstractFloat}(c::Vector{T}, p::GLP,
                                    features::Vector{T})
   W, b, _ = split_theta!(p)
   A_mul_B!(c, W, features)
   c .+= b
end

# noise is in c vector already
function getaction!{T<:AbstractFloat}(c::Vector{T}, p::GLP,
                                      features::Vector{T})
   W, b, ls = split_theta!(p)
   @. c = c * exp(ls) + b
   # c += A*features
   DT = eltype(p.theta)
   if DT == T
      LinAlg.gemv!('N', DT(1.0), W, features, DT(1.0), c)
   else
      c += W*features
   end
end

function loglikelihood!{T<:AbstractFloat}(ll,
                                          theta::Vector{T}, p::GLP,
                                          features::Matrix{T},
                                          actions::Matrix{T})
   W = reshape( theta[p.ranges[:W]],  p.m, p.n)
   b = reshape( theta[p.ranges[:b]],  p.m)
   ls= reshape( theta[p.ranges[:ls]], p.m)

   dt = eltype(theta)
   terms = dt(-0.5*p.m*log(2pi)) - sum(ls)

   mu = broadcast(+, W*features, b) # W*features + b

   zs = ((actions-mu) ./ exp.(ls)).^2
   ll .= sum(zs, 1)
   ll .= ll.*dt(-0.5) .+ terms
end

function loglikelihood(theta, p::GLP, features, actions)
   W = reshape( theta[p.ranges[:W]],  p.m, p.n)
   b = reshape( theta[p.ranges[:b]],  p.m)
   ls= reshape( theta[p.ranges[:ls]], p.m)

   dt = eltype(theta)
   #terms = dt(-0.5*p.m*log(2pi)) - sum(ls)
   terms = -0.5*p.m*log(2pi) - sum(ls)

   mu = broadcast(+, W*features, b) # W*features + b

   zs = ((actions-mu) ./ exp.(ls)).^2
   ll = sum(zs, 1)
   #ll .= ll.*dt(-0.5) .+ terms
   ll .= ll.*-0.5 .+ terms

   return ll
end

function gradLL!{T<:AbstractFloat}(gradll::Matrix{T},
                                   p::GLP,
                                   features::Matrix{T},
                                   actions::Matrix{T})
   #gradloglikelihood(theta) = loglikelihood(theta, p, features, actions)
   #f_jc = ForwardDiff.JacobianConfig(gradloglikelihood, p.theta)
   #@time ForwardDiff.jacobian!(gradll, gradloglikelihood, p.theta, f_jc)

   #function ll_vec{GT<:PolType}(theta::GT, ff::GT, aa::GT)
   function ll_vec(theta, ff, aa)
      loglikelihood(theta, p, ff, aa)
   end

   const n = p.n
   const m = p.m
   const N = size(features, 2) # N = T*numT

   BLAS.set_num_threads(1)
   nthread = min(N, Threads.nthreads())
   Threads.@threads for tid=1:nthread
   #   for tid=1:nthread 
      thread_range = Distributed.splitrange(N, nthread)[tid]

      const ff = zeros(T, n)
      const aa = zeros(T, m)

      const ctp = ReverseDiff.GradientTape(ll_vec, (p.theta, ff, aa))

      gr = map(similar, (p.theta, ff, aa))
      for i=thread_range
         ff .= features[:,i] # implicit conversion and memory saving
         aa .= actions[:,i]

         ReverseDiff.gradient!(gr, ctp, (p.theta, ff, aa)) # taped 
         gradll[:,i] .= gr[1][:]
      end
   end
   BLAS.set_num_threads(Threads.nthreads())
end

function split_theta!(p::GLP)
   w_r = p.ranges[:W]
   b_r = p.ranges[:b]
   l_r = p.ranges[:ls]

   W   = reshape( view(p.theta, w_r), p.m, p.n)
   b   = reshape( view(p.theta, b_r), p.m)
   ls  = reshape( view(p.theta, l_r), p.m)

   return W, b, ls
end

function split_theta(p::GLP)
   w_r = p.ranges[:W]
   b_r = p.ranges[:b]
   l_r = p.ranges[:ls]

   W   = reshape( p.theta[w_r], p.m, p.n)
   b   = reshape( p.theta[b_r], p.m)
   ls  = reshape( p.theta[l_r], p.m)


   return W, b, ls
end

