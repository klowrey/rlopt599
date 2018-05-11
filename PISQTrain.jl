
# DOCOPT faster than argparse, but a little sensitive to spacing
doc = """We use this to find an MPPI trajectory. UW CSE Motion Control Lab.

Usage:
   PISQTrain.jl [(<experiment> <name>)]
   PISQTrain.jl -h | --help

Options:
   -h --help   Show this screen.

"""

using DocOpt
opt = docopt(doc, ARGS)


@everywhere push!(LOAD_PATH, "./")

using Common: plotlines

prefix="/tmp/"
modeldir="$(pwd())/models"
meshfile = "$(modeldir)/meshes/" # hack for now...

expfile=opt["<experiment>"]
expname=opt["<name>"]
myseed = 12345

@eval @everywhere const SEED = $myseed
@everywhere push!(LOAD_PATH, "$(pwd())")

println("Experiment called $(expname)")
println("Loading experiment parameters from $(expfile)")

################################################### start work by loading files
using mjWrap
using MPPI
using ExpFunctions
using LearningStrategies

@eval @everywhere const mjw = mjWrap # scoping issues...

const METHOD = :MPPI
include("$(pwd())/$(expfile)") # LOADING TASK FILE with PARAMETERS
@assert isdefined(:mppi)
@assert isdefined(:mppi_specs)
@assert isdefined(:myfuncs)

# plot stoc & mean every other iter
evaluatepolicy = IterFunction((model,i)->push!(model.trace[:evalscore], myfuncs.evaluate(model.mjsys, model.samples)))

const NITER = 100
plotR = IterFunction(NITER, (model,i)->plotlines(i,"Costs",
                                                 (model.trace[:meanR],"Min"),
                                                 (model.trace[:stocR],"Avg")))

plotEval = IterFunction(NITER, (model,i)->plotlines(i,"Evaluation",
                                                    (model.trace[:evalscore],"")))

plotCTRL = IterFunction(NITER, (model,i)->plotlines(i,"Ctrl",
                                                    model.samples.ctrl[:,:,1]))

plotState = IterFunction(NITER, (model,i)->plotlines(i,"State",
                                                     model.traj.state[1:3,1:i,1])) # X axis plot for now

# meta strategies: order matters!!
mppi_strat = strategy(ControlExp(prefix, expfile,
                                 expname, meshfile), # setup some things, hook for save
                      MPPIRollout(myfuncs),
                      mppi_specs,                  # main loop

                      #evaluatepolicy,             # hooks
                      
                      plotR,            

                      #plotEval,
                      plotCTRL,

                      #plotState,
                      MaxIter(mppi.niter)) #Verbose(MaxIter(mppi.niter)))

@time learn!(mppi, mppi_strat)


########################## done


