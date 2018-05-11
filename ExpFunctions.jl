__precompile__()

module ExpFunctions

struct FunctionSet
   setmodel!::Function
   initstate!::Function
   setcontrol!::Function
   observe!::Function
   reward::Function
   evaluate::Function
end
export FunctionSet

using Common: makesavedir, save
using mjWrap: save_traj
using Policy: save_pol
using LearningStrategies
import LearningStrategies: setup!, hook, cleanup!
export setup!, hook, cleanup!

#export FunctionSet
export PolicyExp
export ControlExp

# Policy Gradient experiment specific things
mutable struct PolicyExp <: LearningStrategy
   prefix::String
   expfile::String
   expname::String
   meshfile::String
   dir::String
   statefile::String
   policyfile::String
   expresults::String

   function PolicyExp(prefix::String, expfile::String,
                      expname::String, meshfile::String)
      return new(prefix, expfile, expname, meshfile, "", "", "", "")
   end
end

function setup!(s::PolicyExp, model)
   info("SETTING UP NPG EXPERIMENT")

   poltype = Base.datatype_name(typeof(model.policy))
   dir = "$(s.prefix)/$(s.expname)_$poltype"
   expdir = makesavedir(dir, s.expfile, model.mjsys.name, s.meshfile)
   s.dir        = "$(expdir)"
   s.statefile  = "$(expdir)/data.jld"
   s.policyfile = "$(expdir)/policy.jld"
   s.expresults = "$(expdir)/expmt.jld"
end

function hook(s::PolicyExp, model, i)
   if i==1 || model.trace[:stocR][i] >= maximum(model.trace[:stocR][1:i-1])
      info("\tSaving traj for experiment ", s.dir)
      save_traj(model.mjsys, model.samples, s.statefile)
      save_pol(model.policy,
               model.mjsys.skip,
               basename(model.mjsys.name),
               s.policyfile) # needs skip and model for policy playback (FIX LATER)
      save(s.expresults, model.trace)
   end
end

function cleanup!(s::PolicyExp, model)
   save(s.expresults, model.trace)
   info("Saved Experiment to ", s.dir)
   # email myself when done
   #run(`bash ./plot2html.sh $(scores[end]) $(expfile)`)
   #if cluster == 1
   #   println(workers())
   #   for w in workers()
   #      rmprocs(w)
   #   end
   #end
end


##### Control Experiment Functions
mutable struct ControlExp <: LearningStrategy
   prefix::String
   expfile::String
   expname::String
   meshfile::String
   dir::String
   statefile::String
   #policyfile::String
   expresults::String

   function ControlExp(prefix::String, expfile::String,
                       expname::String, meshfile::String)
      return new(prefix, expfile, expname, meshfile, "", "", "")
   end
end

function setup!(s::ControlExp, model)
   info("SETTING UP NPG EXPERIMENT")

   dir          = "$(s.prefix)/$(s.expname)_MPPI"
   expdir       = makesavedir(dir, s.expfile, model.mjsys.name, s.meshfile)
   s.dir        = "$(expdir)"
   s.statefile  = "$(expdir)/data.jld"
   s.expresults = "$(expdir)/expmt.jld"
end

#function hook(s::ControlExp, model, i)
#   if i==1 || model.stocR[i] >= maximum(model.stocR[1:i-1])
#      info("\tSaving traj for experiment ", s.dir)
#      save_traj(model.mjsys, model.traj, s.statefile)
#      save(s.expresults, Dict(:stocR => model.stocR[1:i],
#                              :meanR => model.meanR[1:i],
#                              :evals => model.evalscore[1:i]))
#   end
#end
#
function cleanup!(s::ControlExp, model)
   save_traj(model.mjsys, model.traj, s.statefile) # save result of MPC anyway
   save(s.expresults, model.trace)
   info("Saved Experiment to ", s.dir)

   # email myself when done
   #run(`bash ./plot2html.sh $(scores[end]) $(expfile)`)
   #if cluster == 1
   #   println(workers())
   #   for w in workers()
   #      rmprocs(w)
   #   end
   #end
end

end

