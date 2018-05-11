# primary features:
# rollout sequence of controls
# df/dx, df/du derivatives

# features to add:
# address mjdata, mjmodel fields by string or symbol
#   replicate the get_sensors stuff
__precompile__()

module mjWrap

using MuJoCo
using JLD


immutable mjSet
   m::mj.jlModel
   d::mj.jlData
   datas::Vector{mj.jlData}

   # helpers
   sensor_names::Dict{Symbol, Range}
   name::String
   skip::Integer
   mode::Integer

   nq::Integer
   nv::Integer
   nu::Integer
   ns::Integer
   dt::Float64

   #function mjSet(model_name::String) # TODO?
   #end
end

immutable TrajSamples
   T::Integer
   numT::Integer
   s0::Vector{Float64}
   ctrl::Array{Float64,3}
   state::Array{Float64,3}
   obs::Array{Float64,3}
   reward::Array{Float64,2}
end

#=
type TrajDerivatives
   T::Integer
   numT::Integer
   d_x::Array{Float64}
   d_u::Array{Float64}
   s_x::Array{Float64}
   s_u::Array{Float64}

   #c::Array{Float64}
   #c_x::Array{Float64}
   #c_u::Array{Float64}
   #c_xx::Array{Float64}
   #c_uu::Array{Float64}
   #c_ux::Array{Float64}
end
=#

const mjw = mjWrap
export TrajSamples, mjSet, mjw

function finish()
   mj.deactivate()
   return 0;
end

function modelparams(x::mjSet)
   return x.nq, x.nv, x.nu, x.ns
end

function clear_model(x::mjSet)
   mj.deleteModel(x.m.m)
   mj.deleteData(x.d.d)
   for i=1:length(x.datas)
      mj.deleteData(x.datas[i])
   end
end

function load_model(model_name::String, skip::Integer, mode::String="normal", ns::Integer=0)
   pm = mj.loadXML(model_name, C_NULL)
   if pm == nothing
      return
   end
   pd = mj.makeData(pm)

   m, d = mj.mapmujoco(pm, pd)
   ndata = Threads.nthreads()
   datas = Vector{jlData}(ndata)
   print("Making mjDatas: 1 ")
   datas[1] = d
   for i=2:ndata
      print("$i ")
      datas[i] = mj.mapdata(pm, mj.makeData(pm))
   end
   println()

   if mj.get(m, :nsensor) > 0
      sensors = mj.name2range(m, mj.get(m, :nsensor),
                              m.name_sensoradr, m.sensor_adr, m.sensor_dim)
   else
      sensors = Dict(:none=>0:0)
   end

   nq = Int64(mj.get(m, :nq))
   nv = Int64(mj.get(m, :nv))
   nu = Int64(mj.get(m, :nu))
   if ns==0 ns = Int64(mj.get(m, :nsensordata)) end
   dt = mj.get(m, :opt, :timestep)


   return mjSet(m, d, datas, sensors, model_name, skip, 0,
                nq, nv, nu, ns, dt)
end

############ UTILS ############
function wraptopi(ang::Float64)
   return mod(ang+pi, 2pi) - pi
end

function unitfit(ang::Float64, val::Float64)
   clamp.(ang, -val, val)./val
end

function scaledcontrol(m::mj.jlModel, d::mj.jlData,
                       ctrl::AbstractVector{Float64})
   nv = length(d.qvel)
   mark = mj.MARKSTACK(d)

   My = @view d.stack[mark+1:(mark+nv)]
   mj.solveM(m.m, d.d, My, d.qfrc_actuator, nv)

   force = d.qfrc_actuator'*My

   mj.FREESTACK(d, Int32(mark))
   return force
end

############################

function reset(m::mj.jlModel, d::mj.jlData,
               s0::AbstractVector{Float64},
               ctrl0::AbstractVector{Float64},
               obs::AbstractVector{Float64})
   # setup
   nq = length(d.qpos) #mj.get(m, :nq)
   nv = length(d.qvel) #mj.get(m, :nv)

   # reset
   d.qacc .= 0.0
   for i=1:nq
      d.qpos[i] = s0[i]
   end
   for i=1:nv
      d.qvel[i] = s0[nq+i]
   end
   copy!(d.ctrl, ctrl0)
   mj.set(d, :time, 0.0)

   # out
   mj.forward(m, d)
   copy!(obs, d.sensordata)
end

# accepts threading; tid is 1 indexed in Julia
function reset(x::mjSet, tid::Integer,
               s0::AbstractVector{Float64},
               ctrl0::AbstractVector{Float64},
               obs::AbstractVector{Float64})

   reset(x.m, x.datas[tid], s0, ctrl0, obs)
end

function reset(x::mjSet,
               s0::AbstractVector{Float64},
               ctrl0::AbstractVector{Float64},
               obs::AbstractVector{Float64})
   reset(x.m, x.d, s0, ctrl0, obs)
end

########################################################################## step

function quat2euler(quat::Vector{Float64})
   w, x, y, z = quat
   sinr = 2.0 * (w * x + y * z)
   cosr = 1.0 - 2.0 * (x * x + y * y)
   roll = atan2(sinr, cosr)

   # pitch (y-axis rotation)
   sinp = 2.0 * (w * y - z * x)
   if abs(sinp) >= 1
      pitch = copysign(pi / 2, sinp) # use 90 degrees if out of range
   else
      pitch = asin(sinp)
   end

   # yaw (z-axis rotation)
   siny = 2.0 * (w * z + x * y)
   cosy = 1.0 - 2.0 * (y * y + z * z)  
   yaw = atan2(siny, cosy)

   return roll, pitch, yaw
end

function euler2quat(roll::Float64, pitch::Float64, yaw::Float64)
   # blame wikipedia if this doesn't work
   cy = cos(yaw * 0.5)
   sy = sin(yaw * 0.5)
   cr = cos(roll * 0.5)
   sr = sin(roll * 0.5)
   local cp = cos(pitch * 0.5)
   sp = sin(pitch * 0.5)

   w = cy * cr * cp + sy * sr * sp
   x = cy * sr * cp - sy * cr * sp
   y = cy * cr * sp + sy * sr * cp
   z = sy * cr * cp - cy * sr * sp
   return [w, x, y, z]
end

function step!(m::mj.jlModel, d::mj.jlData, skip::Integer,
               ctrl::AbstractVector{Float64},
               s1::AbstractVector{Float64},
               obs::AbstractVector{Float64})
   nq = length(d.qpos)
   copy!(d.ctrl, ctrl)

   for i=1:skip
      mj.step(m, d) # TODO MAKE FASTER with fwd and euler
   end

   mj.forward(m, d)
   s1[1:nq]     = d.qpos
   s1[nq+1:end] = d.qvel
   copy!(obs, d.sensordata)
end

# accepts threading; tid is 1 indexed in Julia
function step!(x::mjSet, tid::Integer,
               ctrl::AbstractVector{Float64},
               s1::AbstractVector{Float64},
               obs::AbstractVector{Float64})
   step!(x.m, x.datas[tid], x.skip, ctrl, s1, obs)
end

# resets position then takes step
function step!(x::mjSet, tid::Integer,
               s0::AbstractVector{Float64},
               ctrl::AbstractVector{Float64},
               s1::AbstractVector{Float64},
               obs::AbstractVector{Float64})
   error("init_step s0 -> s1 not implemented")
end


function step!(x::mjSet,
               ctrl::Array{Float64},
               s1::Array{Float64},
               obs::Array{Float64})
   step!(x.m, x.d, x.skip, ctrl, s1, obs)
end

function alloc_fields(nqnv::Integer, nu::Integer, ns::Integer,
                      T::Integer, numT::Integer)
   ctrl  = zeros(nu, T, numT)
   state = zeros(nqnv, T, numT) # just allocate space
   obs   = zeros(ns,   T, numT)
   return ctrl, state, obs
end

# optionally we can specify what we want our observations to be elsewhere
# instead of in mujoco's sensors format
function allocateTrajSamples(x::mjSet, T::Integer, numT::Integer,
                             ns::Integer=x.ns)
   nq, nv, nu, _ = mjw.modelparams(x)
   ctrl, state, obs = alloc_fields(nq+nv, nu, ns, T, numT)

   s0 = zeros(nq+nv)
   s0[1:nq]     = x.d.qpos
   s0[nq+1:end] = x.d.qvel

   return TrajSamples(T, numT, s0,
                      ctrl, state, obs, zeros(T,numT))
end

#=
function alloc_derivs(m::Model, T::Integer, numT::Integer)
   # to note: this is the format c should expect
   d_x = Array(Float64, m.nqnv, m.nqnv, T, numT)
   d_u = Array(Float64, m.nqnv, m.nu,   T, numT)
   s_x = Array(Float64, m.ns,   m.nqnv, T, numT)
   s_u = Array(Float64, m.ns,   m.nu,   T, numT)
   return TrajDerivatives(T, numT, d_x, d_u, s_x, s_u)
end
=#

function save_traj(x::mjSet,
                   s::TrajSamples,
                   filename::String)
   save(filename,
        "state", s.state,
        "ctrl", s.ctrl,
        "model", basename(x.name),
        "skip", x.skip,
        "mode", "normal")
end

#function rolloutstep(x::mjw.mjSet, 
#                 samp::mjw.TrajSamples,
#                 k::Int, t::int,

function rollout(x::mjw.mjSet, 
                 samp::mjw.TrajSamples,
                 ctrlfunc!::Function=(y...)->nothing,
                 obsfunc!::Function=(y...)->nothing,
                 rewardfunc::Function=(y...)->0.0)
   nthread = Int(min(samp.numT, Int(Threads.nthreads()))) # cant have more threads than data
   BLAS.set_num_threads(1)
   nq, nv, ns, nu = x.nq, x.nv, x.ns, x.nu
   Threads.@threads for tid=1:nthread
      #for tid=1:nthread
      s0, s = Array{Float64}(nq+nv), Array{Float64}(nq+nv)
      o0, o = Array{Float64}(ns), Array{Float64}(ns)
      c     = Array{Float64}(nu)
      thread_range = Distributed.splitrange(samp.numT, nthread)[tid]

      for t=Range(thread_range)
         s[:] = samp.state[:,1,t]
         c[:] = samp.ctrl[:,1,t]
         reset(x, tid, s, c, o)
         obsfunc!(x, tid, s, o)    # arbitrary observation vector manipulation
         samp.state[:,1,t] = s
         s0[:]             = s
         samp.obs[:,1,t]   = o
         o0[:]             = o
         for k=Range(1:samp.T)

            c .= samp.ctrl[:,k,t]  # if ctrlfunc does nothing (no controller)
            # then we will execute the stored control
            ctrlfunc!(c, o)        # we can also set noise to be in ctrl

            step!(x, tid, c, s, o) # get next state, next observation
            obsfunc!(x, tid, s, o) # arbitrary observation vector manipulation

            samp.reward[k,t] = rewardfunc(x, tid,
                                          s0, s,
                                          o0, o, c)
            s0[:] = s
            o0[:] = o
            samp.ctrl[:,k,t] = c
            if k < samp.T
               samp.state[:,k+1,t] = s
               samp.obs[:,k+1,t]   = o
            end
         end
      end
   end
   BLAS.set_num_threads(Threads.nthreads())
end


#=
function rollout(x::mjw.mjSet, 
   samp::mjw.TrajSamples,
   obsfunc!::Function=(y...)->nothing,
   rewardfunc!::Function=(y...)->nothing)
   nthread = min(samp.numT, Threads.nthreads()) # cant have more threads than data
   #Threads.@threads for tid=1:nthread
      for tid=1:nthread
         s0, s = Array{Float64}(x.nq+x.nv), Array{Float64}(x.nq+x.nv)
         o0, o = Array{Float64}(x.ns), Array{Float64}(x.ns)
         c     = Array{Float64}(x.nu)
         thread_range = Distributed.splitrange(samp.numT, nthread)[tid]

         for t=thread_range
            #state= view(samp.state, :, :, t)
            #ctrl = view(samp.ctrl, :, :, t)
            #obs  = view(samp.obs, :, :, t)
            s[:] = samp.state[:,1,t]
            c[:] = samp.ctrl[:,1,t]
            mjw.reset(x, tid, s, c, o)
            obsfunc!(x, s, o) # arbitrary observation vector manipulation
            samp.state[:,1,t] = s
            samp.obs[:,1,t]   = o
            for k=1:samp.T

               c[:] = samp.ctrl[:,k,t]
               mjw.step!(x, tid, c, s, o) # get next state, next observation
               obsfunc!(x, s, o) # arbitrary observation vector manipulation

               samp.reward[k,t] = rewardfunc(x, tid,
               s0, s,
               o0, o, c)
               s0[:] = s
               o0[:] = o
               if k < samp.T
                  samp.state[:,k+1,t] = s
                  samp.obs[:,k+1,t]   = o
               end
            end
         end
      end
   end
   =#

end # mj module


