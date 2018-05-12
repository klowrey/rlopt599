# rlopt

This a Julia version of NPG and MPPI for development and exploration of RL algorithms for UW CSE 599G1.

Organization consists of a number of experiments in their own directories that define the parameters for different algorithsm as well as environment functions, such as observations, rewards, etc. These experiment files are loaded into a training file, either PGTrain.jl or PISQTrain.jl, that uses these parameters to set up the learning loop, before executing the loop.

Currently, this code is somewhat linked to MuJoCo and continuous control tasks, but merely as a means of sampling and environment and storing state/action data. If you would like to implement a wrapper for your own environment, you can inspect mjWrap.jl for potential structure.

If you would like to create new MuJoCo environments, you can either specify observations through sensors in the model xml file, or by manually pre-specifying a observation vector during the creationg of your mjSet struct, and filling it in your designed observation function. You can look at the Sawyer experiment for an example.

A number of management and utility functions are provided in the additional files.

# Install and Setup

To install we need to install some system things in Linux or OSX.

```bash
# LINUX
sudo apt install build-essentials hdf5-tools libglfw3-dev libglew-dev
```

```bash
# macOS
xcode-select --install
brew install hdf5 glfw3 glew
```

Then install Julia by downloading the correct software package from [their website](https://julialang.org/downloads/). A safe place you can extract to on Linux is `/opt/julia` or whereever else you have access; OSX will install to Applications.

You may also want to add the following to your `~/.bashrc` if you're on Linux or `~/.bash_profile` if on OSX:

```bash
export JULIA_HOME=/opt/julia/bin                     # OSX installs to /Applications/julia-0.6.app/Contents/Resources/julia/bin
export PATH=$JULIA_HOME:$PATH                        # if you'd like to have general access to julia

export JULIA_NUM_THREADS=4                           # or your system's number of CPUs

export MUJOCO_KEY_PATH=/home/$USER/.mujoco/mjkey.txt # install mujoco key here
```

Then we can install the package dependencies for this package by running the rollowing command. This command will also load the packages once to induce precompilation as necessary.

```julia
julia required.jl
```

# Usage

There are two modes of usage: training policies with NPG or optimizing trajectories through MPPI. The following commands will run the optimization, saving the best policy or final trajectory to a directory user /tmp with the name prefixed by the second argument passed on the command line.

```bash
julia -O3 PGTrain.jl swimmer/swimmer.jl test   # NPG
julia -O3 PISQTrain.jl swimmer/swimmer.jl test # MPPI
```

In the Julia REPL, we can then view the output of the training/optimizing by doing the following:

```julia
julia> push!(LOAD_PATH, "./")
julia> using ExpSim

julia> simulate("/tmp/test_GLP") # or other saved directory
```

In the screen that pops up, you can switch between viewing modes with ',' and '.' to see either the passive model used in the experiment, saved trajectories, or the saved policy running in real time. Other commands for the viewer can be found under ExpSim.jl or Sim.jl; you can also interact with the model by clicking on bodies and moving them around.


