@everywhere push!(LOAD_PATH, "../")
@everywhere push!(LOAD_PATH, "./")


include("swimmerfunctions.jl")

####################################### Experiment Parameters / HyperParameters 

model_file = "$(modeldir)/swimmer.xml"

# simulator parameters
T = 500
numT = 32
skip = 5
niter = 30

# solver parameters
gamma = 0.995
gae = 0.98

fullFIM = true

norm_step_size = 0.05

cg_iter= 12
cg_reg = 1e-6
cg_tol = 1e-10

#### deterministic randomization
srand(12345) # random seed, set globally

dtype = Float64

my_mjsys       = mjw.load_model(model_file, skip, "normal")

if METHOD == :NPG ######################################### NPG startup
   #my_policy      = Policy.GLP{dtype}(my_mjsys.ns, my_mjsys.nu) # inputs: n, m
   my_policy      = Policy.NN{dtype}(my_mjsys.ns, my_mjsys.nu, 32, 2) # inputs: n, m

   pgmodel        = PolicyGradModel(copy(my_policy.theta),
                                    my_mjsys,
                                    my_policy,
                                    T, numT, 10, niter)

   # NPG baseline aka value function approximation
   baseline    = Baseline.Quadratic{Float64}(1e-5, my_mjsys.ns, T, numT)
   #baseline    = Baseline.NN{Float64}(my_mjsys.ns, T, numT, 128)

   pg_specs   = NPGStrategy{dtype}(pgmodel,
                                   baseline,
                                   fullFIM,
                                   norm_step_size,
                                   gamma, gae,
                                   cg_iter, cg_reg, cg_tol)

elseif METHOD == :MPPI ######################################### MPPI startup
   mpcT = 100
   mppi = MPPIModel(eye(my_mjsys.nu)*0.1, # sigma
                    0.3,                  # lambda
                    my_mjsys,
                    mpcT, # horizon for mpc; set to T for non-MPC mode
                    T,
                    numT;
                    theta = randn(my_mjsys.nu, mpcT),
                    s0 = [my_mjsys.d.qpos; my_mjsys.d.qvel], # set initial state here
                   )

   mppi_specs = MPPIStrategy(mppi)
else
   error("WHAT YOU DOING")
end
