ENV["JULIA_CUDA_MEMORY_LIMIT"] = 10^10 # cap the amount of allocated GPU memory, in bytes.


using CUDA, OrdinaryDiffEq, DiffEqFlux, LinearAlgebra, DiffEqSensitivity#, Flux
using Images, Plots, ArgParse
using DifferentialEquations, DelimitedFiles, Logging
# using Parameters: @with_kw
# julia runSim.jl -s 19 -d 19SourcesRdm
# CUDA.usage_limit[] = 10^10  # For a 12GB K80 card
# CUDA.reclaim()


function parseCommandLine()

        # initialize the settings (the description is for the help screen)
        s = ArgParseSettings(description = "Input to generate dataset")

        @add_arg_table! s begin
           # "--opt1"               # an option (will take an argument)
           # "--opt2", "-o"         # another option, with short form
            "--startIndex", "-i"                 # a positional argument
                help = "start index"
                arg_type=Int
                default = 1
            "--endIndex", "-e"
                help = "end Index"
                arg_type=Int
                default = 10000
            "--dir", "-d"
                help = "directory"
                arg_type=String
                default = "/TwoSourcesRdm"
            "--numSources", "-s"
                help = "number of sources"
                arg_type=Int64
                default = 1
            "--radius", "-r"         # another option, with short form
                       help = "source radius"
                       arg_type = Int
                       default = 5
             "--gridSize", "-g"         # another option, with short form
                        help = "Grid Size"
                        arg_type = Int
                        default = 512
              "--pathOut", "-o"         # another option, with short form
                         help = "pathOut"
                         arg_type = String
                         default = "/raid/javier/Datasets/DiffSolver" #pwd()
              "--gpu", "-u"
                        help = "Use GPU?"
                        arg_type = Bool
                        default = false
              "--log", "-l"
                        help = "Log output?"
                        arg_type = Bool
                        default = false
            # "arg5"
                # arg_type=Int
            # "arg6"
                # arg_type=Int=
        end

        return parse_args(s) # the result is a Dict{String,Any}
end


function gen_op(s, use_gpu)
    N=s
    Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
    My = copy(Mx)
    # Mx[2,1] = 2.0
    # Mx[end-1,end] = 2.0
    # My[1,2] = 2.0
    # My[end,end-1] = 2.0
    if use_gpu
        return CuArray(Float32.(Mx)), CuArray(Float32.(My))
    else
        return Mx, My
    end
end

function f(u,p,t)
  α₁ = p
  A = u
  gMyA =  gMy * A
  gAMx = A * gMx
  gDA =  @. Dd*(gMyA + gAMx)
  # dA = @. gDA + α1*BPatch - r1*A*BPatch
  dA = @. gDA - r1*A + α₁
  return dA
end

function fCPU(u,p,t)
  α₁ = p
  A = u
  gMyA =  gMy * A
  gAMx = A * gMx
  gDA =  @. Dd*(gMyA + gAMx)
  # dA = @. gDA + α1*BPatch - r1*A*BPatch
  dA = @. gDA - r1*A + α₁
  return Array(dA)
end

function min_dist_to_border(source, l)
    minimum([source[1], abs(source[1]-l), source[2], abs(source[2]-l)])
end

in_source(x,xs,y,ys,r) = sqrt((x-xs)^2 + (y-ys)^2) <= r


function genInitialCond(s, radius, use_gpu; ns=1)
  ar = [[i,j] for i in 1:s, j in 1:s]
  dist = zeros(ns)
  init_cond = zeros(s,s)
  for ns0 in 1:ns
    source = [rand(radius:s-radius), rand(radius:s-radius)]
    dist[ns0] = min_dist_to_border(source, s)
    init_cond = init_cond .+ [in_source(ar[i,j][1], source[1], ar[i,j][2], source[2], radius) for i in 1:s, j in 1:s] .* rand(0.01:0.0000001:1.0)
  end

  if use_gpu
      CuArray{Float32,2}(init_cond/maximum(init_cond)), dist
  else
      init_cond/maximum(init_cond), dist
  end
#   CuArray{Float32,2}(min.(init_cond,1.0)), dist
end

function genTuples(idx, in_cond, dist, use_gpu; s=100, radius=5, tmax=3000.0,
            dir = "newdata")
  α₁ = in_cond
  u0 = α₁
  if use_gpu
      prob = ODEProblem{false}(f,u0,(0.0,tmax),α₁)
#       @time sol = solve(prob,ROCK2(),progress=true,save_everystep=true,save_start=true)
    sol = solve(prob,ROCK2(),progress=true,save_everystep=true,save_start=true)
  else
      prob = ODEProblem{false}(fCPU,u0,(0.0,tmax),α₁)
#       @time sol = solve(prob,ROCK2(),save_everystep=false,save_start=false, abstol = 1e-3, reltol = 1e-3)
    sol = solve(prob,ROCK2(),save_everystep=false,save_start=false, abstol = 1e-3, reltol = 1e-3)
  end
  out = Array(sol[end])/maximum(sol[end])    #radius^2

  path = PATH * dir * "/test2"   # temp: remove test2
  isdir(path) || mkpath(path)
  writedlm(path * "/Cell_$idx.dat", reshape(Array(α₁),:))
  writedlm(path * "/Field_$idx.dat", reshape(out,:))
  # writedlm(path * "/Dist_$idx.dat", dist)
  Array(α₁), out
end

function runFunction(start, stop, dir; ns=1, gridSize=100, radius = 5)
  for i in start:stop
    @info i
    in_cond, dist = genInitialCond(gridSize, r, use_gpu; ns=ns)
    init_cond, final_state = genTuples(i, in_cond, dist, use_gpu; dir=dir)
    # f1 = heatmap(init_cond)
    # f2 = heatmap(final_state)
    # fig = plot(f1,f2, layout = @layout grid(2,1))
    # display(fig)
    CUDA.reclaim()
    if log
        flush(io)
    end
  end
end

function myLog(ns, init, finish, log)
    isdir(pwd() * "/Log/") || mkdir(pwd() * "/Log/")
    io = open(pwd() * "/Log/" * "log_$(ns)_$(init)_$(finish).log", "w+")
    return io
end


# parsed_args = parseCommandLine()
# st = parsed_args["startIndex"]
# stop = parsed_args["endIndex"]
# # dir_name = "dataset-Test" #parsed_args["dir"]
# dir_name = parsed_args["dir"]
# num_sources = parsed_args["numSources"]

# # PATH = "/N/u/jtoledom/Carbonate/Data/"
# PATH = parsed_args["pathOut"] #"/scratch/st-mgorges-1/jtoledom/nobackup/DiffSolver/" #
# r = parsed_args["radius"]
# gridSize = parsed_args["gridSize"]
# use_gpu = parsed_args["gpu"]
# log = parsed_args["log"]


if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parseCommandLine()
    st = parsed_args["startIndex"]
    stop = parsed_args["endIndex"]
    # dir_name = "dataset-Test" #parsed_args["dir"]
    dir_name = parsed_args["dir"]
    num_sources = parsed_args["numSources"]

    # PATH = "/N/u/jtoledom/Carbonate/Data/"
    PATH = parsed_args["pathOut"] #"/scratch/st-mgorges-1/jtoledom/nobackup/DiffSolver/" #
    r = parsed_args["radius"]
    gridSize = parsed_args["gridSize"]
    use_gpu = parsed_args["gpu"]
    log = parsed_args["log"]
    
    if log
        io = myLog(num_sources, st, stop, log)
        logger = SimpleLogger(io)
        global_logger(logger)
    end

    @info "Logging output? $(log)"
    @info "Using GPUs? $(use_gpu)"

    α₁, _ = genInitialCond(gridSize, r, use_gpu)
    gMx, gMy = gen_op(size(α₁,1), use_gpu)        # This generates the discrete laplacian as in CR's tutorial
    r1 = 1.0/400.0
    Dd = 1.0
    if use_gpu
        gMyA = CuArray(Float32.(zeros(size(gMx))))
        gAMx = CuArray(Float32.(zeros(size(gMx))))
        gDA = CuArray(Float32.(zeros(size(gMx))))
    else
        gMyA = zeros(size(gMx))
        gAMx = zeros(size(gMx))
        gDA = zeros(size(gMx))
    end
    u0 = α₁

    runFunction(st, stop, dir_name; ns=num_sources, gridSize=gridSize, radius = r)

    if log
        close(io)
    end
    
end