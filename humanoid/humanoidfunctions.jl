
using Distances

using mjWrap
using MuJoCo

################################################## Experiment Centric Functions
## accepts model and trajectory data
## returns sum of trajectory rewards for each traj; saves per timestep reward
function rewardfunction(x::mjWrap.mjSet, tid::Int,
                        s0::AbstractVector{Float64}, # pass in states
                        s1::AbstractVector{Float64},
                        o0::AbstractVector{Float64}, # observations
                        o1::AbstractVector{Float64},
                        ctrl::AbstractVector{Float64})::Float64
   d = x.datas[tid]

   reward = 2
   reward -= 1*s1[1]         #(10*s1[1])^2
   reward -= 1*s1[2]         #(10*s1[2])^2
   reward -= 4*abs(1.1 - s1[3]) #(10*(1.1 - s1[3]))^2

   reward -= 1e-7*mjWrap.scaledcontrol(x.m, d, d.qfrc_actuator)
   #reward -= 1e-2*norm(ctrl)^2

   #reward -= 1e-7*norm(o1[4:end]-o0[4:end])^2 # min jerk
   #reward -= 1e-6*norm(o1-o0)^2 # min jerk
   
   return reward
end

function obsfunction(x::mjWrap.mjSet, tid::Int,
                     s::AbstractVector{Float64},
                     o::AbstractVector{Float64})
   #o[1:x.nq] .= 
   #o .= s
   d = x.datas[tid]
   o .= d.qacc
end

## accepts model,
## default state,
## preallocated matrix of states (nqnv x numT)
function initfunction!(x::mjWrap.mjSet,
                       s::mjWrap.TrajSamples)
   mag = 0.2
   init_state = view( s.state, :, 1, : )
   for t=1:s.numT
      @. init_state[:,t] = s.s0 + rand() * mag - mag/2.0
      init_state[3,t] -= 0.1
   end
end

function ctrlfunction!(x::mjWrap.mjSet, ctrl::Array{Float64})
   randn!(ctrl)
end

# modelfunc should takes in time index (iteration index) to know change curric
function modelfunction!(x::mjWrap.mjSet, iter::Int=0)
end

## evaluation function: reward function shapes exploration, evaluation
## function defines success or failure
function evalfunction(x::mjWrap.mjSet, d::mjWrap.TrajSamples)
   return mean(d.state[3,end,:]) # x position at time T, for all numT
end

using ExpFunctions
myfuncs = FunctionSet(modelfunction!,
                      initfunction!,
                      ctrlfunction!,
                      obsfunction,
                      rewardfunction,
                      evalfunction)


