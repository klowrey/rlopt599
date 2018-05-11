
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
   #hand = d.site_xpos[:,1]
   #goal = d.site_xpos[:,2]
   #ball = d.site_xpos[:,3]

   d_hand = o1[27] #norm(hand-ball)^2
   d_goal = o1[28] #norm(goal-ball)^2

   #reward = 1.0 - 3.0*d_goal - d_hand - 1e-1*norm(ctrl)^2
   # kendall
   #if d_hand >= 0.02
   #   reward = 1.0 - 1.0*d_hand
   #else
   #   reward = 2.0 - 1.0*d_goal
   #   if d_goal < 0.02
   #      reward += 3.0
   #   end
   #end
   ##reward -= 1e-1*norm(ctrl)^2
   #reward -= 1e-2*norm(d.qvel)^2

   #reward = 5 - 1*d_hand - 10*d_goal
   reward = -1*d_hand - 1*d_goal
   #if d_hand < 0.005
   #   reward += 5
   #end
   #if d_goal < 0.005
   #   reward += 10
   #end
   #reward -= 1e-2*norm(d.qvel)^2
   reward -= 1e-2*norm(ctrl)^2

   #mark = mj.MARKSTACK(x.datas[tid])
   #My = @view d.stack[mark+1:(mark+x.nv)]
   #mj.solveM(x.m.m, x.datas[tid].d, My, x.datas[tid].qfrc_actuator, x.nv)
   #reward -= 1e-4*(x.datas[tid].qfrc_actuator'*My)
   #mj.FREESTACK(x.datas[tid], Int32(mark))
   
   return reward #max(reward, -3.0)
end

function obsfunction(x::mjWrap.mjSet, tid::Int,
                     s::AbstractVector{Float64},
                     o::AbstractVector{Float64})
   hand = x.datas[tid].site_xpos[:,1]
   goal = x.datas[tid].site_xpos[:,2]
   ball = x.datas[tid].site_xpos[:,3]

   d_hand = norm(hand-ball)^2
   d_goal = norm(goal-ball)^2

   # magnet for ball to hand
   o[31:33] .= 0.0 #cosine_dist(hand, goal) 
   if d_hand < 0.005
      x.datas[tid].qpos[11:13] .= hand # set ball qpos to be hand's
      s[11:13] .= hand # ALSO MODIFY THE STATE
      o[31:33] .= 1.0
   end

   o[1:13] = x.datas[tid].qpos
   o[14:26] .= x.datas[tid].qvel./10.0
   o[27] = d_hand
   o[28] = d_goal
   o[29] = 0.0 #cosine_dist(hand, ball)
   o[30] = 0.0 #cosine_dist(hand, goal) 
end

## accepts model,
## default state,
## preallocated matrix of states (nqnv x numT)
function initfunction!(x::mjWrap.mjSet,
                       s::mjWrap.TrajSamples)
   s0 = s.s0
   s0[1:x.nq] = [0.369814, 0.40224, -1.18184, -1.50206, 1.46349, -0.215438, -0.000427812, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   mag = 0.2
   z = 0.6
   rot = pi #2pi #2*pi
   init_state = view( s.state, :, 1, : )
   init_state .= 0.0
   for t=1:s.numT
      init_state[1:x.nq,t] = s0[1:x.nq] + rand(x.nq) * mag - mag/2.0

      # goal
      init_state[ 8:10,t] = z*rand(3) - z/2
      init_state[   10,t] += 0.3

      # ball
      init_state[11:13,t] = z*rand(3) - z/2
      init_state[   13,t] += 0.3
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
   return mean(d.obs[28,end,:]) # x position at time T, for all numT
end

using ExpFunctions
myfuncs = FunctionSet(modelfunction!,
                      initfunction!,
                      ctrlfunction!,
                      obsfunction,
                      rewardfunction,
                      evalfunction)


