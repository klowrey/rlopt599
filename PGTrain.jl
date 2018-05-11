
# DOCOPT faster than argparse, but a little sensitive to spacing
doc = """We use this to train Policy Gradients. UW CSE Motion Control Lab.

Usage:
   PGTrain.jl [(<experiment> <name>)]
   PGTrain.jl -h | --help

Options:
   -h --help   Show this screen.

"""

using DocOpt
opt = docopt(doc, ARGS)


@everywhere push!(LOAD_PATH, "./")

using Common

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
using Policy
using PolicyGradient
using ExpFunctions
using LearningStrategies

const METHOD = :NPG
include("$(pwd())/$(expfile)") # LOADING TASK FILE with PARAMETERS
@assert isdefined(:pgmodel)
@assert isdefined(:pg_specs)
@assert isdefined(:myfuncs)


# PG specific rollouts
policyrollout = PolicyRollout(myfuncs) # in setup! this makes the stochastic and mean functions

# plot stoc & mean every other iter
# TODO Generall strategies; put somewhere else?
evaluatepolicy = IterFunction((model,i)->push!(model.trace[:evalscore], myfuncs.evaluate(model.mjsys, model.meansamples)))

const NITER = 4
plotR = IterFunction(NITER, (model,i)->Common.plotlines(i,"Reward",
                                                        (model.trace[:stocR],"Stoc"),
                                                        (model.trace[:meanR],"Mean")))

plotEval = IterFunction(NITER, (model,i)->Common.plotlines(i,"Evaluation",
                                                           (model.trace[:evalscore],"")))

plotCTRL = IterFunction(NITER, (model,i)->
                        begin
                           _, idx = findmax(sum(model.meansamples.reward, 1))
                           Common.plotlines(i,"Ctrl", model.meansamples.ctrl[:,:,idx])
                        end )
#plotCTRL2 = IterFunction(NITER, (model,i)->Common.plotlines(i,"Ctrl Stoc", model.samples.ctrl[:,:,1]))

# meta strategies: order matters!!
pg_strat = strategy(PolicyExp(prefix, expfile,
                              expname, meshfile), # setup some things, hook for save

                    policyrollout,           # stochastic and mean rollouts
                    pg_specs,              # main loop for NPG
                    evaluatepolicy,          # hooks
                    plotR,            
                    plotEval,
                    plotCTRL,
                    #plotCTRL2,
                    Verbose(MaxIter(pgmodel.niter)))

@time learn!(pgmodel, pg_strat)


########################## done
