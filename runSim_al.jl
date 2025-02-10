#!/usr/bin/env julia

ENV["JULIA_CUDA_MEMORY_LIMIT"] = 10^10

using ArgParse
using HDF5
using CUDA
using OrdinaryDiffEq
using LinearAlgebra
using DifferentialEquations
using DelimitedFiles
using Logging
using Dates

function parseCommandLine()
    s = ArgParseSettings(description = "Diffusion PDE solver reading source parameters from HDF5")
    @add_arg_table! s begin
        "--startIndex", "-i"
            help = "start index"
            arg_type=Int
            default = 1
        "--endIndex", "-e"
            help = "end index"
            arg_type=Int
            default = 1000
        "--dir", "-d"
            help = "Output directory name"
            arg_type=String
            default = "SimResults"
        "--paramFile", "-p"
            help = "Path to HDF5 file containing the source parameters"
            arg_type=String
            default = "my_source_params.h5"
        "--pathOut", "-o"
            help = "Directory path where to save results"
            arg_type = String
            default = "/eagle/RECUP/twang/rose/diffusion_solver/simulation_results/"
        "--gpu", "-u"
            help = "Use GPU?"
            arg_type = Bool
            default = false
        "--log", "-l"
            help = "Log output?"
            arg_type = Bool
            default = true
    end
    return parse_args(s)
end

function gen_op(s, use_gpu)
    N = s
    Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
    My = copy(Mx)

    # Possibly for boundary, but those are commented out
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
    A  = u
    gMyA = gMy * A
    gAMx = A * gMx
    gDA  = @. Dd * (gMyA + gAMx)
    # dA = @. gDA + α1*BPatch - r1*A*BPatch
    dA   = @. gDA - r1 * A + r1 * α₁
    return dA
end

function fCPU(u,p,t)
    α₁ = p
    A  = u
    gMyA = gMy * A
    gAMx = A * gMx
    gDA  = @. Dd * (gMyA + gAMx)
    # dA = @. gDA + α1*BPatch - r1*A*BPatch
    dA   = @. gDA - r1 * A + r1 * α₁
    return Array(dA)
end

# This is not used in any function, do we need to keep them?
function min_dist_to_border(source, l)
    minimum([source[1], abs(source[1]-l), source[2], abs(source[2]-l)])
end

in_source(x,xs,y,ys,r) = sqrt((x-xs)^2 + (y-ys)^2) <= r

function genInitialCond(s, positions, amplitudes, radius, ns, use_gpu)
    #FIXME: Need an info for value
    ar = [[i,j] for i in 1:s, j in 1:s]
    dist = zeros(ns)
    init_cond = zeros(s,s)
    for ns0 in 1:ns
        source = [positions[k, 1], positions[k, 2]]
        dist[ns0] = min_dist_to_border(source, s)
        init_cond = init_cond .+ [in_source(ar[i,j][1], source[1], ar[i,j][2], source[2], radius) for i in 1:s, j in 1:s] .* amplitudes[k]
    end

    if use_gpu
        return CuArray{Float32,2}(init_cond/maximum(init_cond)), dist
    else
        return init_cond/maximum(init_cond), dist
    end
end

function genTuples(idx, in_cond, dist, use_gpu; tmax=3000.0, dir="newdata")
    α₁ = in_cond
    u0 = α₁
    if use_gpu
        prob = ODEProblem{false}(f, u0, (0.0,tmax), α₁)
#       @time sol = solve(prob,ROCK2(),progress=true,save_everystep=true,save_start=true)
        sol  = solve(prob, ROCK2(), progress=true, save_everystep=true, save_start=true)
    else
        prob = ODEProblem{false}(fCPU, u0, (0.0,tmax), α₁)
#       @time sol = solve(prob,ROCK2(),save_everystep=false,save_start=false, abstol = 1e-3, reltol = 1e-3)
        sol  = solve(prob, ROCK2(), save_everystep=false, save_start=false, abstol=1e-3, reltol=1e-3)
    end
    out = Array(sol[end]) / maximum(sol[end])
  
    path = PATH * dir * "/test2"
    isdir(path) || mkpath(path)
    writedlm(path * "/Cell_$idx.dat", reshape(Array(α₁), :))
    writedlm(path * "/Field_$idx.dat", reshape(out, :))
    # writedlm(path * "/Dist_$idx.dat", dist)
  
    return Array(α₁), out
end

function runFunction(start, stop, param_file, dir, use_gpu)
    h5_data = h5open(param_file, "r") do f
        (
            positions = read(f, "positions"),    # shape (num_samples, ns, 2)
            amplitudes = read(f, "amplitudes"),  # shape (num_samples, ns)
            ns = read(f, "num_sources"),         # shape (num_samples)
            radius = read(f, "radius"),          # shape (1)
            grid_size = read(f, "grid_size")     # shape (1)
        )
    end

    num_samples = size(h5_data.positions, 1)
    grid_size   = h5_data.grid_size[1]
    radius      = h5_data.radius[1]

    if en > num_samples
        error("Requested endIndex=$en is greater than available samples=$num_samples.")
    end

    @info "From $start to $stop"
    for i in start:stop
        @info "Sim global index $i / $num_samples"
        pos_i = h5_data.positions[i, :, :]
        amp_i = h5_data.amplitudes[i, :]
        ns_i  = h5_data.ns[i]

        in_cond, dist = genInitialCond(grid_size, pos_i, amp_i, radius, ns_i, use_gpu)
        init_cond, final_state = genTuples(i, in_cond, dist, use_gpu; dir=dir)
        # f1 = heatmap(init_cond)
        # f2 = heatmap(final_state)
        # fig = plot(f1,f2, layout = @layout grid(2,1))
        # display(fig)
        if use_gpu
            CUDA.reclaim()
        end
        if log
            flush(io)
        end
    end
end

function myLog(startIdx, endIdx, do_log)
    isdir(pwd() * "/Log/") || mkdir(pwd() * "/Log/")
    io = open(pwd() * "/Log/" * "log_$(startIdx)_$(endIdx).log", "w+")
    return io
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parseCommandLine()

    st        = parsed_args["startIndex"]
    ed        = parsed_args["endIndex"]
    paramFile = parsed_args["paramFile"]
    dir_name  = parsed_args["dir"]
    PATH      = parsed_args["pathOut"]
    use_gpu   = parsed_args["gpu"]
    do_log    = parsed_args["log"]

    if do_log
        io = myLog(st, ed, do_log)
        logger = SimpleLogger(io)
        global_logger(logger)
    end

    @info "$(now()) Logging? $(do_log)"
    @info "$(now()) Using GPUs? $(use_gpu)"

    grid_size = h5open(param_file, "r") do f
        read(f, "grid_size")[1]
    end

    gMx, gMy = gen_op(grid_size, use_gpu)
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

    runFunction(st, ed, param_file, dir_name, use_gpu)

    if do_log
        close(io)
    end
end

