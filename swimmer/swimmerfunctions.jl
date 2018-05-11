
using mjWrap

################################################## Experiment Centric Functions
## accepts model and trajectory data
## returns sum of trajectory rewards for each traj; saves per timestep reward
function rewardfunctionvec(x::mjWrap.mjSet,
                                    s::mjWrap.TrajSamples)::Vector{Float64}
    vel_x  = diff(s.state[1,:,:]) / (x.dt*x.skip)
    angle  = -1e-1 * abs.(s.obs[1,:,:])
    c_ctrl = -1e-4 * sum( s.ctrl.*s.ctrl, 1 )[1,:,:]
    s.reward[:,:] = c_ctrl + angle
    s.reward[2:end,:] += vel_x
    return sum(s.reward, 1)[1,:]
end

function rewardfunction(x::mjWrap.mjSet, tid::Int,
                        s0::AbstractVector{Float64}, # pass in states
                        s1::AbstractVector{Float64},
                        o0::AbstractVector{Float64}, # observations
                        o1::AbstractVector{Float64},
                        ctrl::AbstractVector{Float64})::Float64

   pos    = Float64( (s1[1] - s0[1]) / (x.dt*x.skip) ) # forward x direction
   angle  = -1e-1 * abs( o1[1] )
   c_ctrl = -1e-3 * norm(ctrl)^2

   reward = 2*pos + c_ctrl + angle
   return reward
end

function obsfunction(x::mjWrap.mjSet, tid::Int,
                                 s::AbstractVector{Float64},
                                 o::AbstractVector{Float64})
    o[1] = mjWrap.wraptopi(o[1])
end

## accepts model,
## default state,
## preallocated matrix of states (nqnv x numT)
function initfunction!(x::mjWrap.mjSet,
                                   s::mjWrap.TrajSamples)
    s0 = s.s0
    mag = 0.2
    rot = pi #2pi #2*pi
    init_state = view( s.state, :, 1, : )
    for t=1:s.numT
        init_state[:,t] = s0 + rand(x.nq+x.nv) * mag - mag/2.0
        #init_state[3,t] = s0[3] + rand() * rot - rot/2.0 # diverse
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
    # swimmer success: max distance along x axis covered at end of T
    xaxis_index = 1
    return mean(d.state[xaxis_index,end,:]) # x position at time T, for all numT
end

using ExpFunctions
myfuncs = FunctionSet(modelfunction!,
                      initfunction!,
                      ctrlfunction!,
                      obsfunction,
                      rewardfunction,
                      evalfunction)

