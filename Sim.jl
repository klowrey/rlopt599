__precompile__()

module Sim

import GLFW
using MuJoCo
using StaticArrays

##################################################### globals
const fontscale = mj.FONTSCALE_150 # can be 100, 150, 200

mutable struct Simulation
   # visual interaction controls
   lastx::Float64
   lasty::Float64
   button_left::Bool
   button_middle::Bool
   button_right::Bool

   lastbutton::Int32
   lastclicktm::Float64

   refreshrate::Int

   showhelp::Int
   showoption::Bool
   showdepth::Bool
   showfullscreen::Bool
   showsensor::Bool
   slowmotion::Bool

   showinfo::Bool
   paused::Bool
   keyreset::Int

   framenum::Int
   lastframenum::Int

   # MuJoCo things
   scn::Ptr{mj.mjvScene}
   cam::Ptr{mj.mjvCamera}
   vopt::Ptr{mj.mjvOption}
   pert::Ptr{mj.mjvPerturb}
   con::Ptr{mj.mjrContext}
   figsensor::Ptr{mj.mjvFigure}
   m::mj.jlModel
   d::mj.jlData

   # GLFW handle
   window::GLFW.Window

   function Simulation(m::mj.jlModel, d::mj.jlData, window::GLFW.Window)
      new(0.0, 0.0, false, false, false, 0, 0.0,
          0, 0, false, false, false, false, false, true, true, 0, 0, 0,
          Ptr{mj.mjvScene}(Base.Libc.malloc(sizeof(mj.mjvScene))),
          Ptr{mj.mjvCamera}(Base.Libc.malloc(sizeof(mj.mjvCamera))),
          Ptr{mj.mjvOption}(Base.Libc.malloc(sizeof(mj.mjvOption))),
          Ptr{mj.mjvPerturb}(Base.Libc.malloc(sizeof(mj.mjvPerturb))),
          Ptr{mj.mjrContext}(Base.Libc.malloc(sizeof(mj.mjrContext))),
          Ptr{mj.mjvFigure}(Base.Libc.malloc(sizeof(mj.mjvFigure))),
          m, d, window)
   end
end

export Simulation

const keycmds = Dict{GLFW.Key, Function}(GLFW.KEY_F1=>(s)->begin  # help
                                         s.showhelp += 1
                                         if s.showhelp > 2 s.showhelp = 0 end
                                      end,
                                      GLFW.KEY_F2=>(s)->begin  # option
                                         s.showoption = !s.showoption;
                                      end,
                                      GLFW.KEY_F3=>(s)->begin  # info
                                         s.showinfo = !s.showinfo;
                                      end,
                                      GLFW.KEY_F4=>(s)->begin  # depth
                                         s.showdepth = !s.showdepth;
                                      end,
                                      GLFW.KEY_F5=>(s)->begin  # toggle full screen
                                         s.showfullscreen = !s.showfullscreen;
                                         s.showfullscreen ? GLFW.MaximizeWindow(s.window):GLFW.RestoreWindow(s.window)
                                      end,
                                      #GLFW.KEY_F6=>(s)->begin  # stereo
                                      #   s.stereo = mj.get(s.scn, :stereo) == mj.mjSTEREO_NONE ? mjSTEREO_QUADBUFFERED : mj.mjSTEREO_NONE
                                      #   mj.set(s.scn, :stereo)
                                      #end,
                                      GLFW.KEY_F7=>(s)->begin  # sensor figure
                                         s.showsensor = !s.showsensor;
                                      end,
                                      GLFW.KEY_F8=>(s)->begin  # profiler
                                         s.showprofiler = !s.showprofiler;
                                      end,
                                      GLFW.KEY_ENTER=>(s)->begin  # slow motion
                                         s.slowmotion = !s.slowmotion;
                                         s.slowmotion ? println("Slow Motion Mode!"):println("Normal Speed Mode!")
                                      end,
                                      GLFW.KEY_SPACE=>(s)->begin  # pause
                                         s.paused = !s.paused
                                         s.paused ? println("Paused"):println("Running")
                                      end,
                                      GLFW.KEY_PAGE_UP=>(s)->begin    # previous keyreset
                                         s.keyreset = min(mj.get(s.m.m, :nkey) - 1, s.keyreset + 1)
                                      end,
                                      GLFW.KEY_PAGE_DOWN=>(s)->begin  # next keyreset
                                         s.keyreset = max(-1, s.keyreset - 1)
                                      end,
                                      # continue with reset
                                      GLFW.KEY_BACKSPACE=>(s)->begin  # reset
                                         mj.resetData(s.m.m, s.d.d)
                                         if s.keyreset >= 0 && s.keyreset < mj.get(s.m.m, :nkey)
                                            mj.set(s.d, :time, s.m.key_time[s.keyreset+1])
                                            s.d.qpos[:] = s.m.key_qpos[:,s.keyreset+1]
                                            s.d.qvel[:] = s.m.key_qvel[:,s.keyreset+1]
                                            s.d.act[:]  = s.m.key_act[:,s.keyreset+1]
                                         end
                                         mj.forward(s.m, s.d)
                                         #profilerupdate()
                                         sensorupdate(s)
                                      end,
                                      GLFW.KEY_RIGHT=>(s)->begin  # step forward
                                         if s.paused
                                            mj.step(s.m, s.d)
                                            #profilerupdate()
                                            sensorupdate(s)
                                         end
                                      end,
                                      GLFW.KEY_LEFT=>(s)->begin  # step back
                                         if s.paused
                                            dt = mj.get(s.m, :opt, :timestep)
                                            mj.set(s.m, :opt, :timestep, -dt)
                                            #cleartimers(s.d);
                                            mj.step(s.m, s.d);
                                            mj.set(s.m, :opt, :timestep, dt)
                                            #profilerupdate()
                                            sensorupdate(s)
                                         end
                                      end,
                                      GLFW.KEY_DOWN=>(s)->begin  # step forward 100
                                         if s.paused
                                            #cleartimers(d);
                                            for n=1:100 mj.step(s.m, s.d) end
                                            #profilerupdate();
                                            sensorupdate(s)
                                         end
                                      end,
                                      GLFW.KEY_UP=>(s)->begin  # step back 100
                                         if s.paused
                                            dt = mj.get(s.m, :opt, :timestep)
                                            mj.set(s.m, :opt, :timestep, -dt)
                                            #cleartimers(d)
                                            for n=1:100 mj.step(s.m, s.d) end
                                            mj.set(s.m, :opt, :timestep, dt)
                                            #profilerupdate();
                                            sensorupdate(s)
                                         end
                                      end,
                                      GLFW.KEY_ESCAPE=>(s)->begin  # free camera
                                         mj.set(s.cam, :type, Int(mj.CAMERA_FREE))
                                      end,
                                      GLFW.KEY_EQUAL=>(s)->begin  # bigger font
                                         if fontscale < 200
                                            fontscale += 50
                                            mj.mjr_makeContext(s.m.m, s.con, fontscale)
                                         end
                                      end,
                                      GLFW.KEY_MINUS=>(s)->begin  # smaller font
                                         if fontscale > 100
                                            fontscale -= 50;
                                            mj.mjr_makeContext(s.m.m, s.con, fontscale);
                                         end
                                      end,
                                      GLFW.KEY_LEFT_BRACKET=>(s)->begin  # '[' previous fixed camera or free
                                           fixedcam = mj.get(s.cam, :type)
                                           if mj.get(s.m.m, :ncam) > 0 && fixedcam == Int(mj.CAMERA_FIXED)
                                              fixedcamid = mj.get(s.cam, :fixedcamid)
                                              if (fixedcamid  > 0)
                                                 mj.set(s.cam, :fixedcamid, fixedcamid-1)
                                              else
                                                 mj.set(s.cam, :type, Int(mj.CAMERA_FREE))
                                              end
                                           end
                                        end,
                                        GLFW.KEY_RIGHT_BRACKET=>(s)->begin  # ']' next fixed camera
                                           if mj.get(s.m.m, :ncam) > 0
                                              fixedcam = mj.get(s.cam, :type)
                                              fixedcamid = mj.get(s.cam, :fixedcamid)
                                              if fixedcam != Int(mj.CAMERA_FIXED)
                                                 mj.set(s.cam, :type, Int(mj.CAMERA_FIXED))
                                              elseif fixedcamid < mj.get(s.m.m, :ncam) - 1
                                                 mj.set(s.cam, :fixedcamid, fixedcamid+1)
                                              end
                                           end
                                        end,
                                        GLFW.KEY_SEMICOLON=>(s)->begin  # cycle over frame rendering modes
                                           frame = mj.get(s.vopt, :frame)
                                           mj.set(s.vopt, :frame, max(0, frame - 1))
                                        end,
                                        GLFW.KEY_APOSTROPHE=>(s)->begin  # cycle over frame rendering modes
                                           frame = mj.get(s.vopt, :frame)
                                           mj.set(s.vopt, :frame,
                                                  min(Int(mj.NFRAME)-1, frame+1))
                                        end,
                                        GLFW.KEY_PERIOD=>(s)->begin  # cycle over label rendering modes
                                           label = mj.get(s.vopt, :label)
                                           mj.set(s.vopt, :label, max(0, label-1))
                                        end,
                                        GLFW.KEY_SLASH=>(s)->begin  # cycle over label rendering modes
                                           label = mj.get(s.vopt, :label)
                                           mj.set(s.vopt, :label,
                                                  min(Int(mj.NLABEL)-1, label+1))
                                        end)

##################################################### functions
function autoscale(s::Simulation)
   # autoscale
   center = mj.get(s.m, :stat, :center)
   mj.set(s.cam, :lookat, center[1], 1)
   mj.set(s.cam, :lookat, center[2], 2)
   mj.set(s.cam, :lookat, center[3], 3)
   mj.set(s.cam, :distance, 1.5*mj.get(s.m, :stat, :extent))

   # set to free camera
   mj.set(s.cam, :_type, Int(mj.CAMERA_FREE))
end

# init sensor figure
function sensorinit(s::Simulation)
   # set figure to default
   mj.mjv_defaultFigure(s.figsensor)

   # set flags
   mj.set(s.figsensor, :flg_extend, Cint(1))
   mj.set(s.figsensor, :flg_barplot, Cint(1))

   # title
   title = Vector{UInt8}("Sensor data")
   for i=1:length(title)
      mj.set(s.figsensor, :title, title[i], i)
   end

   # y-tick nubmer format
   format = "%.0f"
   for i=1:length(format)
      mj.set(s.figsensor, :yformat, format[i], i)
   end

   # grid size
   mj.set(s.figsensor, :gridsize, 2, 1)
   mj.set(s.figsensor, :gridsize, 3, 2)

   # minimum range
   mj.set(s.figsensor, :range,  0, 1, 1)
   mj.set(s.figsensor, :range,  0, 1, 2)
   mj.set(s.figsensor, :range, -1, 2, 1)
   mj.set(s.figsensor, :range,  1, 2, 2)
end

# update sensor figure
function sensorupdate(s::Simulation)
   const maxline = 10

   for i=1:maxline # clear linepnt
      mj.set(s.figsensor, :linepnt, Cint(0), i)
   end

   lineid = 1 # start with line 0
   m = s.m
   d = s.d

   # loop over sensors
   for n=1:mj.get(m, :nsensor)
      # go to next line if type is different
      if (n > 1 && m.sensor_type[n] != m.sensor_type[n - 1])
         lineid = min(lineid+1, maxline)
      end

      # get info about this sensor
      cutoff = m.sensor_cutoff[n] > 0 ? m.sensor_cutoff[n] : 1.0
      adr = m.sensor_adr[n]
      dim = m.sensor_dim[n]

      # data pointer in line
      p = mj.get(s.figsensor, :linepnt, lineid)

      # fill in data for this sensor
      for i=0:(dim-1)
         # check size
         if ((p + 2i) >= Int(mj.MAXLINEPNT) / 2) break end

         x1 = 2p + 4i + 1
         x2 = 2p + 4i + 3
         mj.set(s.figsensor, :linedata, adr+i, lineid, x1)
         mj.set(s.figsensor, :linedata, adr+i, lineid, x2)

         y1 = 2p + 4i + 2
         y2 = 2p + 4i + 4
         se = d.sensordata[adr+i+1]/cutoff
         mj.set(s.figsensor, :linedata,  0, lineid, y1)
         mj.set(s.figsensor, :linedata, se, lineid, y2)
      end

      # update linepnt
      mj.set(s.figsensor, :linepnt,
             min(Int(mj.MAXLINEPNT)-1, p+2dim),
             lineid)
   end
end

# show sensor figure
function sensorshow(s::Simulation, rect::mj.mjrRect)
   # render figure on the right
   viewport = mj.mjrRect(rect.width - rect.width / 4,
                         rect.bottom,
                         rect.width / 4,
                         rect.height / 3)
   mj.mjr_figure(viewport, s.figsensor, s.con)
end

##################################################### callbacks
function mykeyboard(s::Simulation, window::GLFW.Window,
                    key::GLFW.Key, scancode::Int32, act::GLFW.Action, mods::Int32)
   # do not act on release
   if act == GLFW.RELEASE return end

   try
      keycmds[key](s) # call anon function in Dict with s struct passed in
   catch
      # control keys
      if mods & GLFW.MOD_CONTROL > 0
         if key == GLFW.KEY_A
            autoscale(s)
         elseif key == GLFW.KEY_P
            println(s.d.qpos)
         #elseif key == GLFW.KEY_L && lastfile[0]
         #   loadmodel(window, s.)
         elseif key == GLFW.KEY_Q
            GLFW.SetWindowShouldClose(window, true)
         end
      end

      # toggle visualiztion flag
      flags = mj.get(s.vopt, :flags)
      for i=1:Int(mj.NVISFLAG)
         if (key == Int(mj.VISSTRING[i,3][1]))
            mj.set(s.vopt, :flags, flags[i]==0x00?0x01:0x00, i)
         end
      end
      # toggle rendering flag
      flags = mj.get(s.scn, :flags)
      for i=1:Int(mj.NRNDFLAG)
         if (key == Int(mj.RNDSTRING[i,3][1]))
            mj.set(s.scn, :flags, flags[i]==0x00?0x01:0x00, i)
         end
      end
      # toggle geom/site group
      for i=1:Int(mj.NGROUP)
         if key == i + Int('0')
            sitegroup = mj.get(s.vopt, :sitegroup)
            geomgroup = mj.get(s.vopt, :geomgroup)
            if mods & GLFW.MOD_SHIFT == true
               mj.set(s.vopt, :sitegroup, sitegroup[i]>0?0:1, i)
            else
               mj.set(s.vopt, :geomgroup, geomgroup[i]>0?0:1, i)
            end
         end
      end
   end
end

function mouse_move(s::Simulation, window::GLFW.Window,
                    xpos::Float64, ypos::Float64)
   # no buttons down: nothing to do
   if !s.button_left && !s.button_middle && !s.button_right
      return
   end

   # compute mouse displacement, save
   dx = xpos - s.lastx
   dy = ypos - s.lasty
   s.lastx = xpos
   s.lasty = ypos

   width, height = GLFW.GetWindowSize(window)

   mod_shift = GLFW.GetKey(window, GLFW.KEY_LEFT_SHIFT) || GLFW.GetKey(window, GLFW.KEY_RIGHT_SHIFT)

   # determine action based on mouse button
   if s.button_right
      action = mod_shift ? Int(mj.MOUSE_MOVE_H) : Int(mj.MOUSE_MOVE_V)
   elseif s.button_left
      action = mod_shift ? Int(mj.MOUSE_ROTATE_H) : Int(mj.MOUSE_ROTATE_V)
   else
      action = Int(mj.MOUSE_ZOOM)
   end

   # move perturb or camera
   pert = unsafe_load(s.pert)
   if pert.active != 0
      mj.mjv_movePerturb(s.m.m, s.d.d, action,
                         dx / height, dy / height,
                         s.scn, s.pert);
   else
      mj.mjv_moveCamera(s.m.m, action,
                        dx / height, dy / height,
                        s.scn, s.cam)
   end
end

# past data for double-click detection
function mouse_button(s::Simulation, window::GLFW.Window,
                      button::Int32, act::GLFW.Action, mods::Int32)
   # update button state
   s.button_left = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_LEFT)
   s.button_middle = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_MIDDLE)
   s.button_right = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_RIGHT)

   # Alt: swap left and right
   if mods == GLFW.MOD_ALT
      tmp = s.button_left
      s.button_left = s.button_right
      s.button_right = tmp

      if button == GLFW.MOUSE_BUTTON_LEFT
         button = GLFW.MOUSE_BUTTON_RIGHT;
      elseif button == GLFW.MOUSE_BUTTON_RIGHT
         button = GLFW.MOUSE_BUTTON_LEFT;
      end
   end

   # update mouse position
   x, y = GLFW.GetCursorPos(window)
   s.lastx = x
   s.lasty = y

   # set perturbation
   newperturb = 0;
   pert = unsafe_load(s.pert)
   cam = unsafe_load(s.cam)
   if act == GLFW.PRESS && mods == GLFW.MOD_CONTROL && pert.select > 0 
      # right: translate;  left: rotate
      if s.button_right
         newperturb = Int(mj.PERT_TRANSLATE)
      elseif s.button_left
         newperturb = Int(mj.PERT_ROTATE)
      end
      # perturbation onset: reset reference
      if newperturb>0 && pert.active==0
         mj.mjv_initPerturb(s.m.m, s.d.d, s.scn, s.pert)
      end
   end
   mj.set(s.pert, :active, newperturb)

   # detect double-click (250 msec)
   if act == GLFW.PRESS && (time() - s.lastclicktm < 0.25) && (button == s.lastbutton)
      # determine selection mode
      if button == GLFW.MOUSE_BUTTON_LEFT
         selmode = 1;
      elseif mods == GLFW.MOD_CONTROL
         selmode = 3;
      else
         selmode = 2;
      end
      # get current window size
      width, height = GLFW.GetWindowSize(window)

      # find geom and 3D click point, get corresponding body
      selpnt = zeros(3)
      selgeom = mj.mjv_select(s.m.m, s.d.d, s.vopt,
                              width / height, x / width,
                              (height - y) / height, 
                              s.scn, selpnt)
      selbody = (selgeom >= 0 ? s.m.geom_bodyid[selgeom+1] : 0) # 0 indexed

      # set lookat point, start tracking is requested
      if selmode == 2 || selmode == 3
         # copy selpnt if geom clicked
         # TODO hacks
         if selgeom >= 0
            mj.mju_copy3(cam.lookat, @SVector [selpnt[1], selpnt[2], selpnt[3]])
         end

         # switch to tracking camera
         if selmode == 3 && selbody >= 0
            #cam.trackbodyid = selbody
            #cam.fixedcamid = -1
            mj.set(s.cam, :_type, Int(mj.CAMERA_TRACKING))
            mj.set(s.cam, :trackbodyid, selbody)
            mj.set(s.cam, :fixedcamid, -1)
         end
      else # set body selection
         if selbody >= 0
            # compute localpos
            tmp = SVector{3, Float64}(0.0, 0.0, 0.0)
            #mju_sub3(tmp, selpnt, d->xpos + 3 * pert.select);
            tmp = selpnt - s.d.xpos[:,selbody+1]
            #mju_mulMatTVec(pert.localpos, d->xmat + 9 * pert.select, tmp, 3, 3);
            res = reshape(s.d.xmat[:,selbody+1], 3, 3)' * tmp
            mj.set(s.pert, :localpos, SVector{3}(res))

            # record selection
            mj.set(s.pert, :select, selbody)
         else
            mj.set(s.pert, :select, 0)
         end
      end

      # stop perturbation on select
      mj.set(s.pert, :active, 0)
   end
   # save info
   if act == GLFW.PRESS
      s.lastbutton = button
      s.lastclicktm = time()
   end
end

function scroll(s::Simulation, window::GLFW.Window,
                xoffset::Float64, yoffset::Float64)
   # scroll: emulate vertical mouse motion = 5% of window height
   mj.mjv_moveCamera(s.m.m, Int(mj.MOUSE_ZOOM),
                     0.0, -0.05 * yoffset, s.scn, s.cam);

end

function drop(window::GLFW.Window,
              count::Int, paths::String)
end


function start(mm::mj.jlModel, dd::mj.jlData,
               width=1200, height=900) # TODO named args for callbacks
   # TODO make simulation
   s = Simulation(mm, dd, GLFW.CreateWindow(width, height, "Simulate"))

   # Make the window's context current
   GLFW.MakeContextCurrent(s.window)
   GLFW.SwapInterval(1)

   s.refreshrate = GLFW.GetVideoMode(GLFW.GetPrimaryMonitor()).refreshrate
   println("Refresh Rate: $(s.refreshrate)")

   # mujoco setup
   mj.mjv_makeScene(s.scn, 1000)
   mj.mjv_defaultPerturb(s.pert)
   mj.mjv_defaultCamera(s.cam)
   mj.mjv_defaultOption(s.vopt)
   mj.mjr_defaultContext(s.con)
   mj.mjr_makeContext(s.m.m, s.con, Int(fontscale)) # model specific setup

   #profilerinit();
   sensorinit(s)
   #GLFW.SetKeyCallback(s.window, (w,k,sc,a,m)->mykeyboard(s,w,k,sc,a,m))

   GLFW.SetCursorPosCallback(s.window, (w,x,y)->mouse_move(s,w,x,y))
   GLFW.SetMouseButtonCallback(s.window, (w,b,a,m)->mouse_button(s,w,b,a,m))
   GLFW.SetScrollCallback(s.window, (w,x,y)->scroll(s,w,x,y))
   ##GLFW.SetDropCallback(s.window, drop)

   return s
end

#function __init__()
#   atexit(cleanup)
#end
#function cleanup()
#   GLFW.DestroyWindow(window)
#end

end
