pkgs = ["JLD",
        "ReverseDiff",
        "ForwardDiff",
        "Distributions",
        "LearningStrategies",
        "IterativeSolvers",
        "GLFW",
        "DocOpt",
        "UnicodePlots",
        "Flux"]

Pkg.update()
installed = Pkg.installed()

for p in pkgs
   if haskey(installed, p) == false
      Pkg.add(p)
   else
      info("$p installed")
   end
end

if haskey(installed, "MuJoCo") == false
   Pkg.clone("git://www.github.com/klowrey/MuJoCo.jl.git")
   Pkg.build("MuJoCo")
end


Pkg.checkout("LearningStrategies")

using MuJoCo
for p in pkgs
   info("Preloading $(p)")
   eval(parse("using $(p)"))
end
