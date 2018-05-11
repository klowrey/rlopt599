
__precompile__()

module Common

using UnicodePlots
using JLD

function addmachine(name, count, tunnel::Bool=false)
   #topo=:all_to_all
   topo=:master_slave
   println("ADDING $name to cluster")
   if tunnel
      addprocs([(name, count)]; max_parallel=1, topology=topo, tunnel=true)
   else
      addprocs([(name, count)]; max_parallel=1, topology=topo)
   end
   println("WORKERS: ", nworkers())
end

# ornstein-uhlenbeck process for ctrl noise for stochastic search
# μ  = target value of x[end]; get some random target?
# σ  = noise. sure.
# θ  = how quickly to go to μ; fights against σ
# dt = dt of model * skip; dt of environment
# x0 = initial value of x[1]
function ou_process!(x::AbstractMatrix{Float64},
                     σ::Float64=1.0, θ::Float64=2.0,
                     dt::Float64=1/size(x,2))
   nu = size(x,1)
   T  = size(x,2)-1
   W  = zeros(nu)
   x0 = copy(x[:,1])
   μ  = copy(x[:,end])
   for (i,t) in enumerate(0.0:dt:T*dt)
      ex = exp(-θ*t)
      for j=1:nu
         x[j,i] = x0[j]*ex + μ[j]*(1.0-ex) + σ*ex*W[j]
         W[j] += exp(θ*t)*sqrt(dt)*randn()
      end
   end
end

function ou_process!(x::AbstractVector{Float64}, x0::Float64,
                     μ::Float64=(2rand()-1),
                     σ::Float64=1.0, θ::Float64=2.0,
                     dt::Float64=1/length(x))
   T = length(x)-1
   W = 0.0
   for (i,t) in enumerate(0.0:dt:T*dt)
      ex = exp(-θ*t)
      x[i] = x0*ex + μ*(1.0-ex) + σ*ex*W
      W += exp(θ*t)*sqrt(dt)*randn()
   end
end


function plotlines(iter::Integer,title::String,lines::Matrix)
   yin = minimum(lines)
   yax = maximum(lines)
   p = lineplot(lines[1,:],
                xlim=[0, size(lines,2)], 
                ylim=[yin, yax], 
                title=title,
                width=60, height=10)
   for l=2:size(lines, 1)
      lineplot!(p, lines[l,:])
   end

   display(p)
end

function plotlines(iter::Integer,title::String,lines::Vector)
   yin = minimum(lines)
   yax = maximum(lines)
   p = lineplot(lines,
                xlim=[0, size(lines,2)], 
                ylim=[yin, yax], 
                title=title,
                width=60, height=10)

   display(p)
end

function plotlines(iter::Integer,title::String,lines...)
   if length(lines) < 1
      error("plotlines needs data and title")
   end

   p = lineplot(lines[1][1][:],
                title=title,
                width=60, height=10, name=lines[1][2])
   for l in lines[2:end]
      lineplot!(p, l[1][:], name=l[2])
   end

   display(p)
end

function plotexpmt(file)
   d = load(file)
   plotlines(length(d["stocR"]),"Reward",
             (d["stocR"],"Stoc"),
             (d["meanR"],"Mean"))
   return d
end

function makesavedir(dir_name::String, exp_file::String,
                     m_file::String, mesh_file::String="",
                     overwrite::Bool=false)
   if isdir(dir_name) == false
      mkdir(dir_name)
   else
      if overwrite == false
         val = 1
         while isdir("$(dir_name)_$(val)") == true
            val += 1
         end
         dir_name = "$(dir_name)_$(val)"
         mkdir(dir_name)
      end
   end
   # copy file to save directory
   filename = basename(exp_file)
   cp(exp_file, "$(dir_name)/$(filename)", remove_destination=overwrite)
   info("Saving experiment $exp_file")
   open(exp_file) do f
      for l in eachline(f)
         if startswith(l, "include")
            includefile = basename(split(l, "\"")[2])
            info("Including $includefile")
            cp(dirname(exp_file)*"/"*includefile, "$(dir_name)/$(includefile)", remove_destination=overwrite)
         end
      end
   end
   filename = basename(m_file)
   cp(m_file, "$(dir_name)/$(filename)", remove_destination=overwrite)
   info("Model file $m_file")
   if (mesh_file != "")
      if isdir(mesh_file)
         symlink(mesh_file, dir_name*"/meshes")
      else
         symlink(dirname(mesh_file), dir_name*"/meshes")
      end
   end
   return dir_name
end

function save(expresults::String, data::Dict{Symbol, Vector{Float64}})
    jldopen(expresults, "w") do file
        for d in data
           write(file, "$(d[1])", d[2])
        end
    end
end

end
