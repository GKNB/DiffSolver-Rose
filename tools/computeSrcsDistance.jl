using CSV, DataFrames, Plots, DelimitedFiles, Plots.PlotMeasures


# function computeDistancesHistogramTesting(; n = 3, radius = 5, ns = 20)
#    disTotal = []
#     for i in 1:n
#         a, dist = genInitialCond(512, radius, true; ns=ns, pos=[])
#         arr = cropSource(sign.(Array(a)))
#         pos = findall(x-> x==1, arr);
#         dis = [sqrt((pos[i][1]-pos[j][1])^2 + (pos[i][2]-pos[j][2])^2) for i in 1:size(pos,1) for j in i+1:size(pos,1)]
#         disTotal = vcat(disTotal, dis[dis .> 2*radius])
#     end
#     disTotal
# end

function cropSource(arr; nmax=4, thrs=7)
    arr2 = copy(arr);
    pos = findall(x-> x==1,arr);
    for n in 1:nmax
        for i in pos
            try
                if arr[i[1]+1, i[2]+1] + arr[i[1], i[2]+1] + arr[i[1]+1, i[2]] + arr[i[1]+1, i[2]-1] + arr[i[1], i[2]-1] + arr[i[1]-1, i[2]-1] + arr[i[1]-1, i[2]] + arr[i[1]-1, i[2]+1] < thrs
                    arr2[i[1], i[2]] = 0
                end
            catch
                nothing
            end
        end
        arr = copy(arr2)
        pos = findall(x-> x==1,arr);
    end
    arr
end

function computeDistancesHistogram(DS, testCSV; n=10, radius=5)
   disTotal = []
    for i in 1:n
        a = reshape(readdlm(PATH * DS * "/test/" * testCSV[i,1]),512,512);
        dis = computeDistancesInSample(a)
#         disTotal = vcat(disTotal, dis[dis .> 2*radius])
        try
            disTotal = vcat(disTotal, [size(dis[i][dis[i] .> radius],1) > 0 ? sort(dis[i][dis[i] .> radius])[1] : 0.0 for i in 1:size(dis,1)])
        catch
            @warn "In $DS with $i"
            @error sort(dis[i][dis[i] .> radius])
        end
    end
    disTotal
end

function computeDistancesInSample(a)
    arr = cropSource(sign.(Array(a)))
    pos = findall(x-> x==1, arr);
#     dis = [sqrt((pos[i][1]-pos[j][1])^2 + (pos[i][2]-pos[j][2])^2) for i in 1:size(pos,1) for j in i+1:size(pos,1)] 
    dis = [[sqrt((pos[i][1]-pos[j][1])^2 + (pos[i][2]-pos[j][2])^2) for i in 1:size(pos,1)] for j in 1:size(pos,1)]
    dis
end

function SaveStat(DS, testCSV; n=10)
    dis = computeDistancesHistogram(DS, testCSV; n=n)
    writedlm(PATH * DS * "/SrcsDistStat.dat", dis)
    savefig(plot(dis, st=:histogram, margin=10mm, normalized=true), PATH * DS * "/SrcsDistHist.png");
end

PATH = "/raid/javier/Datasets/DiffSolver/"
DS = ["$(i)SourcesRdm" for i in 2:20];

if abspath(PROGRAM_FILE) == @__FILE__
    for i in 1:size(DS,1)
        testCSV = CSV.read(PATH * DS[i] * "/test.csv", DataFrame);
        SaveStat(DS[i], testCSV; n=4000)
    end
end

