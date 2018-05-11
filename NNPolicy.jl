############################################################################ NN
# TODO there is extra hidden layer...

#function getmean(p::NN, features::Vector{Float64})
#   Win, bin, Wh, bh, Wout, bout, _ = split_theta!(p)
#   getmean(Win, bin, Wh, bh, Wout, bout, features)
#end

function getmean!{T<:AbstractFloat}(c::Vector{T}, p::NN,
                                    features::Vector{T})
   Win, bin, Wh, bh, Wout, bout, _ = split_theta!(p)
   #if p.nlayers == 2
   c[:] = getmean(Win, bin, Wh, bh, Wout, bout, features)
end

function getmean(Win, bin, Wh, bh, Wout, bout, features)
   a = tanh.(Win * features + bin)
   nlayers = size(Wh, 3)
   for i=1:nlayers
      a = tanh.(Wh[:,:,i] * a + bh[:,i])
   end
   return Wout * a + bout
end

function getmean2(Win, bin, Wh1, bh1, #Wh2, bh2,
                  Wout, bout, features)
   a = tanh.(Win * features + bin) # m->nhidden
   a[:] = tanh.(Wh1 * a + bh1) # nhidden -> nhidden
   return Wout * a + bout # nhidden -> m 
end

# features = n
function getaction!{T<:AbstractFloat}(c::Vector{T}, p::NN,
                                      features::Vector{T})
   Win, bin, Wh, bh, Wout, bout, ls = split_theta!(p)
   #getmean!(c, p, features)
   @. c = c * exp(ls)
   a = tanh.(Win * features + bin) # m->nhidden
   a[:] = tanh.(Wh[:,:,1] * a + bh[:,1]) # nhidden -> nhidden
   c[:] += Wout * a + bout # nhidden -> m 
end

function split_theta!(p::NN)
   n = p.n
   m = p.m
   nh = p.nhidden
   nlay = p.nlayers

   # win is matrix from n to nhidden
   Win = reshape( view(p.theta, p.ranges[:Win]), nh, n)
   bin = view( p.theta, p.ranges[:bin])

   # nlayers of nhidden*nhidden
   Wh = reshape( view(p.theta, p.ranges[:Wh]), nh, nh, nlay) # W[:,:,i]
   bh = reshape( view(p.theta, p.ranges[:bh]), nh, nlay) # b[:,i]

   # nhidden -> m actions
   Wout = reshape( view(p.theta, p.ranges[:Wout]), m, nh) # W[:,:]
   bout = reshape( view(p.theta, p.ranges[:bout]), m) # b[:]

   ls = reshape( view(p.theta, p.ranges[:ls]), m) # log_std

   return Win, bin, Wh, bh, Wout, bout, ls
end

function loglikelihood!{T<:AbstractFloat}(ll,
                                          theta::Vector{T}, p::NN,
                                          features::Matrix{T},
                                          actions::Matrix{T})
   const N    = size(features, 2) # N = T*numT
   const n    = p.n
   const m    = p.m
   const nh   = p.nhidden
   const rwin = p.ranges[:Win]
   const rbin = p.ranges[:bin]
   const rwh1 = p.ranges[:Wh1]
   const rbh1 = p.ranges[:bh1]
   const rwou = p.ranges[:Wout]
   const rbou = p.ranges[:bout]
   const rls  = p.ranges[:ls]

   const Win  = reshape( view( theta, rwin), nh, n)
   const bin  = view( theta, rbin)
   const Wh1  = reshape( view( theta, rwh1), nh, nh)
   const bh1  = view( theta, rbh1)
   const Wout = reshape( view( theta, rwou), m, nh)
   const bout = view( theta, rbou)
   const ls   = view( theta, rls)

   term1 = -0.5*m*log(2pi)
   term2 = -sum(ls)
   terms = term1+term2

   if size(p.scratch1) != (nh, N) p.scratch1 = zeros(nh, N) end
   if size(p.scratch2) != (nh, N) p.scratch2 = zeros(nh, N) end
   if size(p.scratch3) != (m, N) p.scratch3 = zeros(m, N) end
   
   # hidden 1
   A_mul_B!(p.scratch1, Win, convert(Matrix{T}, features))
   broadcast!(+, p.scratch1, p.scratch1, bin)
   p.scratch1 .= tanh.(p.scratch1)

   A_mul_B!(p.scratch2, Wh1, p.scratch1)
   broadcast!(+, p.scratch2, p.scratch2, bh1)
   p.scratch2 .= tanh.(p.scratch2)

   A_mul_B!(p.scratch3, Wout, p.scratch2)
   broadcast!(+, p.scratch3, p.scratch3, bout)

   p.scratch3 .= ((convert(Matrix{T},actions)-p.scratch3) ./ exp.(ls)).^2
   sum!(ll, p.scratch3)
   ll .= ll.*-0.5 .+ terms
end

function loglikelihood(theta, p::NN, features, actions)
   const n    = p.n
   const m    = p.m
   const nh   = p.nhidden
   const rwin = p.ranges[:Win]
   const rbin = p.ranges[:bin]
   const rwh1 = p.ranges[:Wh1]
   const rbh1 = p.ranges[:bh1]
   const rwou = p.ranges[:Wout]
   const rbou = p.ranges[:bout]
   const rls  = p.ranges[:ls]
   Win = reshape( theta[rwin], nh, n)
   bin = theta[rbin]

   Wh1 = reshape( theta[rwh1], nh, nh)
   bh1 = theta[rbh1]

   Wout = reshape( theta[rwou], m, nh)
   bout = theta[rbou]

   ls = theta[rls]

   dt = eltype(theta)
   #terms = dt(-0.5*p.m*log(2pi)) - sum(ls)
   terms = -0.5*p.m*log(2pi) - sum(ls)

   a = tanh.(broadcast(+, Win * features, bin)) # m->nhidden
   a = tanh.(broadcast(+, Wh1 * a, bh1)) # nhidden -> nhidden
   mu = broadcast(+, Wout * a, bout) # nhidden -> m 

   zs = ((actions-mu) ./ exp.(ls)).^2
   ll = sum(zs, 1)
   ll .= ll.*-0.5 .+ terms
   return ll 
end

function gradLL!{T<:AbstractFloat}(gradll::Matrix{T},
                                   p::NN,
                                   features::Matrix{T},
                                   actions::Matrix{T})
   const n     = p.n
   const m     = p.m
   const nh    = p.nhidden
   const rwin  = p.ranges[:Win]
   const rbin  = p.ranges[:bin]
   const rwh1  = p.ranges[:Wh1]
   const rbh1  = p.ranges[:bh1]
   const rwou  = p.ranges[:Wout]
   const rbou  = p.ranges[:bout]
   const rls   = p.ranges[:ls]
   const N = size(features, 2) # N = T*numT

   function ll_vec(theta, ff, aa)
      loglikelihood(theta, p, ff, aa)
   end

   #println("reverse split")
   BLAS.set_num_threads(1)
   nthread = min(N, Threads.nthreads())
   Threads.@threads for tid=1:nthread
   #@time for tid=1:nthread 
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


