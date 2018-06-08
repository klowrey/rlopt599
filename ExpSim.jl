

__precompile__()

module ExpSim

using Sim
using MuJoCo, Common, GLFW
using Policy: getmean!, AbstractPolicy, loadpolicy
using Flux
using mjWrap #: load_model, mjSet

using JLD
using ExpFunctions

export simulate 

@enum RUN_MODE runmodel=1 rundata=2 runpolicy=3
mutable struct ExpDisplay
   T::Int
   t::Int
   K::Int
   k::Int
   nqnv::Int
   states::Array{Float64, 3}

   pol #::AbstractPolicy
   using_policy::Bool
   skip::Int

   mode::RUN_MODE

   mjsys::mjWrap.mjSet
   myfuncs #::ExpFunctions.FunctionSet

   function ExpDisplay(mjsys, states, pol, skip)
      nqnv, T, K = size(states)
      return new(T, 1, K, 1, nqnv, states,
                 pol, true, skip, runmodel, mjsys, nothing)
   end
end

function traj2state(exd::ExpDisplay)
   t, k = exd.t, exd.k
   m, d = exd.mjsys.m, exd.mjsys.d
   nq = mj.get(m, :nq)
   d.qpos .= exd.states[1:nq, t, k]
   println(d.qpos')
   d.qvel .= exd.states[(nq+1):end, t, k]
   mj.forward(m, d)
end

function file2state(s::Simulation, exd::ExpDisplay) 
   #copy data from traj struct, keeping track of indicies
   if mod1(s.framenum, exd.skip) == exd.skip && s.lastframenum != s.framenum
      if (exd.t >= exd.T) 
         exd.t = 1  #start of new trajectory
         exd.k += 1
         if exd.k > exd.K exd.k = 1 end
         mj.set(s.d, :time, 0.0)
         @printf("Rendering Trajectory %d\n", exd.k)
      end
      traj2state(exd)
      exd.t += 1 # advance our saved trajectory pointer
      s.lastframenum = s.framenum
   end
end

function applypolicy(s::Simulation, exd::ExpDisplay)
   d = exd.mjsys.d
   if mod1(s.framenum, exd.skip) == exd.skip && s.lastframenum != s.framenum
      # Needs to eval the function loaded at run time
      obs = zeros(exd.mjsys.ns)
      eval( :($(exd.myfuncs.observe!)($(exd.mjsys), 1, [$(d.qpos); $(d.qvel)], $(obs))) )
      getmean!(d.ctrl, exd.pol, obs)
      s.lastframenum = s.framenum
   end
end

function my_step(s::Simulation, exd::Union{ExpDisplay,Void})
   m = s.m
   d = s.d
   if exd == nothing
      mj.step(m, d)
   else
      if exd.mode == rundata
         file2state(s, exd)
         t = mj.get(d, :time) + mj.get(m, :opt, :timestep)
         mj.set(d, :time, t) #manually advance time instead of mj.step
      elseif exd.mode == runpolicy && exd.using_policy && exd.pol != nothing
         applypolicy(s, exd)
         mj.step(m, d)
      else # model mode / passive policy mode
         d.ctrl .= 0.0
         mj.step(m, d)
      end
   end
   s.framenum += 1
end

function simulation(s::Simulation, exd::Union{ExpDisplay,Void})
   # println("simulation")
   d = s.d
   m = s.m
   if s.paused
      if mj.get(s.pert, :active) > 0
         mj.mjv_applyPerturbPose(m.m, d.d, s.pert, 1)  # move mocap and dynamic bodies
         mj.forward(m, d)
      end
   else
      #slow motion factor: 10x
      factor = (s.slowmotion ? 10 : 1)

      # advance effective simulation time by 1/refreshrate
      startsimtm = mj.get(d, :time)
      while ((mj.get(d, :time) - startsimtm) * factor < (1.0 / s.refreshrate))
         # clear old perturbations, apply new
         #mju_zero(d->xfrc_applied, 6 * m->nbody);
         d.xfrc_applied .= 0.0
         if mj.get(s.pert, :select) > 0
            mj.mjv_applyPerturbPose(m.m, d.d, s.pert, 0) # move mocap bodies only
            mj.mjv_applyPerturbForce(m.m, d.d, s.pert)
         end

         my_step(s, exd)

         # break on reset
         if (mj.get(d, :time) < startsimtm) break end
      end
   end
end

function render(s::Simulation, exd::Union{ExpDisplay,Void}, w::GLFW.Window)
   wi, hi = GLFW.GetFramebufferSize(w)
   rect = mj.mjrRect(Cint(0), Cint(0), Cint(wi), Cint(hi))
   smallrect = mj.mjrRect(Cint(0), Cint(0), Cint(wi), Cint(hi))

   simulation(s, exd)

   # update scene
   mj.mjv_updateScene(s.m.m, s.d.d,
                      s.vopt, s.pert, s.cam, Int(mj.CAT_ALL), s.scn)
   # render
   mj.mjr_render(rect, s.scn, s.con)

   if s.showsensor
      if (!s.paused) Sim.sensorupdate(s) end
      Sim.sensorshow(s, smallrect)
   end

   if s.showinfo
      str_slow = ""
      if s.slowmotion
         str_slow = "(10x slowdown)"
      end
      if s.paused
         str_paused = "Paused "
      else
         str_paused = "Running "#*str_slow
      end
      if exd != nothing
         if exd.mode == runmodel
            status = str_slow*"\nPassive Model"
         elseif exd.mode == rundata
            str_paused *= "\nTraj\nT\n"
            status = str_slow*"\nData Mode\n$(exd.k)\n$(exd.t)"
         elseif exd.mode == runpolicy
            status = str_slow*"\nPolicy Interaction"
         end
      else
         status = str_slow*"\nPassive Model"
      end

      mj.mjr_overlay(Int(mj.FONT_NORMAL), Int(mj.GRID_BOTTOMLEFT), rect,
                     str_paused,
                     status, s.con)
   end

   # Swap front and back buffers
   GLFW.SwapBuffers(w)
end

function mycustomkeyboard(s::Simulation, exd::Union{ExpDisplay,Void}, window::GLFW.Window,
                          key::GLFW.Key, scancode::Int32, act::GLFW.Action, mods::Int32)
   if act == GLFW.RELEASE return end

   if exd != nothing # more than just model d
      mode = exd.mode
      if key == GLFW.KEY_COMMA
         if mode==runmodel mode=runpolicy else mode=Int(mode) - 1 end
      elseif key == GLFW.KEY_PERIOD
         if mode==runpolicy mode=runmodel else mode=Int(mode) + 1 end
      elseif mode == runpolicy && key == GLFW.KEY_P
         exd.using_policy = !exd.using_policy
         if exd.using_policy println("Using Policy") end
      end
      exd.mode = mode

      if exd.mode == rundata
         if mods & GLFW.MOD_SHIFT > 0
            if key == GLFW.KEY_RIGHT
               exd.k += 1; if exd.k > exd.K exd.k = 1 end
               exd.t = 1
               traj2state(exd)
            elseif key == GLFW.KEY_LEFT
               exd.k -= 1; if exd.k < 1 exd.k = exd.K end
               exd.t = 1
               traj2state(exd)
            end
         else
            if s.paused
               if key == GLFW.KEY_RIGHT
                  exd.t += 1; if exd.t > exd.T exd.t = 1 end
                  traj2state(exd)
               elseif key == GLFW.KEY_LEFT
                  exd.t -= 1; if exd.t < 1 exd.t = exd.T end
                  traj2state(exd)
               end
            end
         end
      end
   end

   Sim.mykeyboard(s, window, key, scancode, act, mods)
end

macro preloadexp(dir)
   return quote
      files        = readdir($dir)
      modelfile    = $dir*"/"*files[find((x)->endswith(x, ".xml"), files)][1]
      functionfile = $dir*"/"*files[find((x)->endswith(x, "functions.jl"), files)][1]
      polfile      = $dir*"/policy.jld"
      if isfile($dir*"/mean.jld")
         datafile  = $dir*"/mean.jld" #TODO HACK loading mean samples for POLO
      else
         datafile  = $dir*"/data.jld"
      end

      #info("Loading Experiment $dir")
      #info("Model file: $modelfile")
      #info("Function file: $functionfile")
      #info("Loading policy file $polfile")
      include(functionfile)

      expmt = Common.plotexpmt($dir*"/expmt.jld") # why not
      maxiter = indmax(expmt["stocR"])

      if isfile(polfile)
         mypolicy, frameskip, _ = loadpolicy(polfile)
      else
         mypolicy = nothing 
         frameskip = 4
         warn("Can't load policy, or no policy to load")
      end
      if isdefined(:ns) == false
         ns = size(load(datafile, "obs"), 1)
      end
      my_mjsys = mjw.load_model(modelfile, frameskip, "normal", ns)

      states = load(datafile, "state")
      exd = ExpDisplay( my_mjsys, load(datafile, "state"), mypolicy, frameskip)

      #exd.myfuncs = Main.myfuncs
      exd.myfuncs = myfuncs
      info("Affecting model accoring to iteration $maxiter")
      for i=1:maxiter
         exd.myfuncs.setmodel!(exd.mjsys, i) # applies curriculum
      end
      return exd
   end
end

function loadexp(dir, width, height)
   info(dir)
   exd = eval( :(@preloadexp($dir)) )

   s = Sim.start(exd.mjsys.m, exd.mjsys.d,  width, height)

   # Simulation struct and ExpDisplay struct share mjmodel and mjdata
   GLFW.SetKeyCallback(s.window, (w,k,sc,a,mo)->mycustomkeyboard(s,exd,w,k,sc,a,mo))
   GLFW.SetWindowRefreshCallback(s.window, (w)->render(s,exd,w))

   return s, exd
end

function loadmodel(modelfile, width, height)
   ptr_m = mj.loadXML(modelfile, C_NULL)
   ptr_d = mj.makeData(ptr_m)
   m, d = mj.mapmujoco(ptr_m, ptr_d)
   s = Sim.start(m, d, width, height)
   info("Model file: $modelfile")

   # Simulation struct and ExpDisplay struct share mjmodel and mjdata
   GLFW.SetKeyCallback(s.window, (w,k,sc,a,mo)->mycustomkeyboard(s,nothing,w,k,sc,a,mo))
   GLFW.SetWindowRefreshCallback(s.window, (w)->render(s,nothing,w))

   return s, nothing
end

function start(args::Vector{String}, width=800, height=480)
   opt = settings(args)

   if isdir(opt["arg1"]) 
      s, exd = loadexp(opt["arg1"], width, height)
   elseif endswith(opt["arg1"], ".xml")
      s, exd = loadmodel(opt["arg1"], width, height)
   else
      error("Unrecognized file input.")
   end

   return s, exd
end

function simulate(s, exd::Union{ExpDisplay,Void})
   # Loop until the user closes the window
   Sim.autoscale(s)
   while !GLFW.WindowShouldClose(s.window)
      render(s, exd, s.window)
      GLFW.PollEvents()
   end
   GLFW.DestroyWindow(s.window)
end

function simulate(f::String, width=800, height=480)
   if isdir(f) 
      s, exd = loadexp(f, width, height)
   elseif endswith(f, ".xml")
      s, exd = loadmodel(f, width, height)
   else
      error("Unrecognized file input.")
   end
   simulate(s, exd)
end

end
