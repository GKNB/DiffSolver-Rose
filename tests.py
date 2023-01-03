import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.utils as vutils
import json
import PIL
import logging
import sys
import argparse
import getpass


sys.path.insert(1, '/home/' + getpass.getuser() +'/Projects/DiffSolver/DeepDiffusionSolver/util')
# sys.path.insert(1, '/home/javier/Projects/DiffSolver/DeepDiffusionSolver/util')

from loaders import generateDatasets, inOut, saveJSON, loadJSON#, MyData
from NNets import SimpleCNN
from tools import accuracy, tools, per_image_error, predVsTarget, errInDS
from plotter import myPlots, plotSamp, plotSampRelative

from trainModel import DiffSur

class thelogger(object):
    def logFunc(self, PATH, dict, dir="0"):
            self.initTime = datetime.now()
            os.path.isdir(PATH + "Logs/") or os.mkdir(PATH + "Logs/")
            os.path.isdir(PATH + "Logs/" + dir) or os.mkdir(PATH + "Logs/" + dir)
            path = PATH + "Logs/" + dir + "/"

            self.logging = logging
            self.logging = logging.getLogger()
            self.logging.setLevel(logging.DEBUG)
            self.handler = logging.FileHandler(os.path.join(path, 'tests.log'))
            self.handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                fmt='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.handler.setFormatter(formatter)
            self.logging.addHandler(self.handler)

            #         self.logging = logging
            #         self.logging.basicConfig(filename=os.path.join(path, 'DiffSolver.log'), level=logging.DEBUG)
            self.logging.info(f'{str(self.initTime).split(".")[0]} - Log')
            
def inverse_huber_loss(target,output, C=0.5):
    absdiff = torch.abs(output-target)
#     C = 0.5#*torch.max(absdiff).item()
#     return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
    return torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) )

class myLoss(object):
    def __init__(self, dict, ep=1, lossSelection="step", wtan=1, w2=6000, alph=2, w=1, delta=0.5):
        self.dict = dict
        self.ep = ep
        
    def forward(self, output, target):
        if lossSelection == "step":
            loss = (1 + torch.tanh(wtan*target) * w2) * torch.abs((output - target)**alph)
        elif lossSelection == "exp":
            loss = torch.exp(-torch.abs(torch.ones_like(output) - output)/w) * torch.abs((output - target)**alph)
        elif lossSelection == "huber":
            loss = (1 + torch.tanh(wtan*target) * w2) * torch.nn.HuberLoss(reduction='none', delta=delta)(output, target)
#         elif lossSelection == "toggle":
#             if np.mod(np.divmod(ep, self.dict["togIdx"])[0], 2) == 0:
#                 loss = (1 + torch.tanh(self.dict["wtan"]*target) * self.dict["w2"]) * torch.abs((output - target)**self.dict["alph"])
#             else:
#                 loss = torch.exp(-torch.abs(torch.ones_like(output) - output)/self.dict["w"]) * torch.abs((output - target)**self.dict["alph"])
#         elif lossSelection == "rand":
# #             r = np.random.rand()            
#             if self.dict["r"][-1]<self.dict["p"]:
#                 loss = (1 + torch.tanh(self.dict["wtan"]*target) * self.dict["w2"]) * torch.abs((output - target)**self.dict["alph"])
#             else:
#                 loss = torch.exp(-torch.abs(torch.ones_like(output) - output)/self.dict["w"]) * torch.abs((output - target)**self.dict["alph"])
        elif lossSelection == "invhuber":
            loss = torch.exp(-torch.abs(torch.ones_like(output) - output)/w) * inverse_huber_loss(target,output, C = delta)
        elif lossSelection == "invhuber2":
            loss = inverse_huber_loss(target,output, C = delta)
        return loss

# /raid/javier/Datasets/DiffSolver/
# '/home/' + getpass.getuser() +'/Projects/DiffSolver/Results/'
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Deep Diffusion Solver")
    parser.add_argument('--path', dest="path", type=str, default='/home/' + getpass.getuser() +'/Projects/DiffSolver/Results/',
                        help="Specify path to dataset")
    parser.add_argument('--dataset', dest="dataset", type=str, default="All",
                        help="Specify dataset")
    parser.add_argument('--dir', dest="dir", type=str, default="100DGX-" + str(datetime.now()).split(" ")[0],
                        help="Specify directory name associated to model")
    parser.add_argument('--bashtest', dest="bashtest", type=bool, default=False,
                        help="Leave default unless testing flow")
    
    parser.add_argument('--bs', dest="bs", type=int, default=50,
                        help="Specify Batch Size")
    parser.add_argument('--nw', dest="nw", type=int, default=8,
                        help="Specify number of workers")
    parser.add_argument('--ngpu', dest="ngpu", type=int, default=1,
                        help="Specify ngpu. (Never have tested >1)")
    parser.add_argument('--lr', dest="lr", type=float, default=0.0001,
                        help="Specify learning rate")
    parser.add_argument('--maxep', dest="maxep", type=int, default=100,
                        help="Specify max epochs")
    
    parser.add_argument('--newdir', dest="newdir", type=bool, default=False,
                        help="Is this a new model?")
    parser.add_argument('--newtrain', dest="newtrain", type=bool, default=False,
                        help="Are you starting training")
    
    
    parser.add_argument('--transformation', dest="transformation", type=str, default="linear",
                        help="Select transformation: linear, sqrt or log?")
    parser.add_argument('--loss', dest="loss", type=str, default="exp",
                        help="Select loss: exp, step, toggle or rand?")
    parser.add_argument('--wtan', dest="wtan", type=float, default=10.0,
                        help="Specify hparam wtan")
    parser.add_argument('--w', dest="w", type=float, default=1.0,
                        help="Specify hparam w")
    parser.add_argument('--alpha', dest="alpha", type=int, default=1,
                        help="Specify hparam alpha")
    parser.add_argument('--w2', dest="w2", type=float, default=4000.0,
                        help="Specify hparam w2")
    parser.add_argument('--delta', dest="delta", type=float, default=0.02,
                        help="Specify hparam delta")
    parser.add_argument('--toggle', dest="toggle", type=int, default=1,
                        help="Specify hparam toggle")
      
    args = parser.parse_args()
    
    
    ###Start Here
    PATH = args.path # "/raid/javier/Datasets/DiffSolver/"
#     DATASETNAME = args.dataset # "All"

    dir = args.dir #'1DGX' #'Test'#str(21)
    BATCH_SIZE=args.bs #50
    NUM_WORKERS=args.nw #8
    ngpu = args.ngpu #1
    lr = args.lr #0.0001
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    diffSolv = DiffSur().to(device)
    os.listdir(PATH + "Dict/")

    
    ##MSE Test
    selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}

    datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
    error, errorField, errorSrc = [], [], []
    
    myLog = thelogger()
    myLog.logFunc(PATH, dict, dir)

    for selectedDir in selectedDirs.keys():
        dir = selectedDir
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#        print(dict, '\n')
        
        myLog.logging.info(f'Generating tests using MSE... for model {selectedDir}')
        try:
            ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
            myLog.logging.info(f'Using Best model!')
        except:
            ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
        theModel.eval();
        
        for (j, ds) in enumerate(datasetNameList):
            myLog.logging.info(f'Dataset: {ds}')
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
    #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=512)
            arr = errInDS(theModel, testloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
            selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
            selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
            selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
            selectedDirs[selectedDir]["max"]["all"].append(arr[3])
            selectedDirs[selectedDir]["max"]["field"].append(arr[4])
            selectedDirs[selectedDir]["max"]["src"].append(arr[5])
            selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
            selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
            selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
            selectedDirs[selectedDir]["min"]["all"].append(arr[9])
            selectedDirs[selectedDir]["min"]["field"].append(arr[10])
            selectedDirs[selectedDir]["min"]["src"].append(arr[11])
            selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
            selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
            selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

            selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
            selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
            selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
            selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
            selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
            selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
            selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
            selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
            selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
            selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
            selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
            selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
            selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
            selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
            selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
        myLog.logging.info(f'Finished tests over datasets')

    modelName = next(iter(selectedDirs.keys()))
    saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_MSE.json')
    myLog.logging.info(f'JSON object saved')
    
    
    ##InvH Test
#     losstest = myLoss(dict,lossSelection=args.loss).forward
#     selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}

#     datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
#     error, errorField, errorSrc = [], [], []

#     for selectedDir in selectedDirs.keys():
#         dir = selectedDir
#         os.listdir(os.path.join(PATH, "Dict", dir))[0]
#         dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
# #        print(dict, '\n')
        
#         myLog.logging.info(f'Generating tests using {args.loss}... for model {selectedDir}')
#         try:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
#             myLog.logging.info(f'Using Best model!')
#         except:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
#         theModel.eval();
        
#         for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
#             trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
#     #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=512)
#             arr = errInDS(theModel, testloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
#             selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
#             selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
#             selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
#             selectedDirs[selectedDir]["max"]["all"].append(arr[3])
#             selectedDirs[selectedDir]["max"]["field"].append(arr[4])
#             selectedDirs[selectedDir]["max"]["src"].append(arr[5])
#             selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
#             selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
#             selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
#             selectedDirs[selectedDir]["min"]["all"].append(arr[9])
#             selectedDirs[selectedDir]["min"]["field"].append(arr[10])
#             selectedDirs[selectedDir]["min"]["src"].append(arr[11])
#             selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
#             selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
#             selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

#             selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
#             selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
#             selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
#             selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
#             selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
#             selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
#             selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
#             selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
#             selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
#             selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
#             selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
#             selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
#             selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
#             selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
#             selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
#     saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_{args.loss}.json')
#     myLog.logging.info(f'JSON object saved')
    
    
#     ##MSE Train
#     selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}

#     datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
#     error, errorField, errorSrc = [], [], []

#     for selectedDir in selectedDirs.keys():
#         dir = selectedDir
#         os.listdir(os.path.join(PATH, "Dict", dir))[0]
#         dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
# #        print(dict, '\n')
#         try:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
#             myLog.logging.info(f'Using Best model!')
#         except:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
#         theModel.eval();
# #         myLog = thelogger()
# #         myLog.logFunc(PATH, dict, dir)
#         myLog.logging.info(f'Generating tests using MSE... for model {selectedDir}')
#         for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
#             trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
#     #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=512)
#             arr = errInDS(theModel, trainloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
#             selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
#             selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
#             selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
#             selectedDirs[selectedDir]["max"]["all"].append(arr[3])
#             selectedDirs[selectedDir]["max"]["field"].append(arr[4])
#             selectedDirs[selectedDir]["max"]["src"].append(arr[5])
#             selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
#             selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
#             selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
#             selectedDirs[selectedDir]["min"]["all"].append(arr[9])
#             selectedDirs[selectedDir]["min"]["field"].append(arr[10])
#             selectedDirs[selectedDir]["min"]["src"].append(arr[11])
#             selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
#             selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
#             selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

#             selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
#             selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
#             selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
#             selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
#             selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
#             selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
#             selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
#             selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
#             selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
#             selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
#             selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
#             selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
#             selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
#             selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
#             selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
#     saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-tr-{modelName}_MSE.json')
#     myLog.logging.info(f'JSON object saved')

###################     ##MAE Test   
    selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}
    datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
    error, errorField, errorSrc = [], [], []

    for selectedDir in selectedDirs.keys():
        dir = selectedDir
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#        print(dict, '\n')
        try:
            ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
            myLog.logging.info(f'Using Best model!')
        except:
            ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
        theModel.eval();
        myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
        for (j, ds) in enumerate(datasetNameList):
            myLog.logging.info(f'Dataset: {ds}')
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
            arr = errInDS(theModel, testloader, device, transformation=dict["transformation"])
            selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
            selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
            selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
            selectedDirs[selectedDir]["max"]["all"].append(arr[3])
            selectedDirs[selectedDir]["max"]["field"].append(arr[4])
            selectedDirs[selectedDir]["max"]["src"].append(arr[5])
            selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
            selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
            selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
            selectedDirs[selectedDir]["min"]["all"].append(arr[9])
            selectedDirs[selectedDir]["min"]["field"].append(arr[10])
            selectedDirs[selectedDir]["min"]["src"].append(arr[11])
            selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
            selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
            selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

            selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
            selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
            selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
            selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
            selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
            selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
            selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
            selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
            selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
            selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
            selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
            selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
            selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
            selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
            selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])            
        myLog.logging.info(f'Finished tests over datasets')

    modelName = next(iter(selectedDirs.keys()))
    saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}.json')
    myLog.logging.info(f'JSON object saved')
    
    
#     ##MAE Train
#     selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}
#     datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
#     error, errorField, errorSrc = [], [], []

#     for selectedDir in selectedDirs.keys():
#         dir = selectedDir
#         os.listdir(os.path.join(PATH, "Dict", dir))[0]
#         dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
# #        print(dict, '\n')
#         try:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
#             myLog.logging.info(f'Using Best model!')
#         except:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
#         theModel.eval();
#         myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
#         for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
#             trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
#             arr = errInDS(theModel, trainloader, device, transformation=dict["transformation"])
#             selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
#             selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
#             selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
#             selectedDirs[selectedDir]["max"]["all"].append(arr[3])
#             selectedDirs[selectedDir]["max"]["field"].append(arr[4])
#             selectedDirs[selectedDir]["max"]["src"].append(arr[5])
#             selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
#             selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
#             selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
#             selectedDirs[selectedDir]["min"]["all"].append(arr[9])
#             selectedDirs[selectedDir]["min"]["field"].append(arr[10])
#             selectedDirs[selectedDir]["min"]["src"].append(arr[11])
#             selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
#             selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
#             selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

#             selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
#             selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
#             selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
#             selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
#             selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
#             selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
#             selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
#             selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
#             selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
#             selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
#             selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
#             selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
#             selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
#             selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
#             selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])            
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
#     saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-tr-{modelName}.json')
#     myLog.logging.info(f'JSON object saved')
    

    myLog.logging.info(f'Generating Sample')
    dsName = "19SourcesRdm"    #args.dataset #"19SourcesRdm"
    trainloader, testloader = generateDatasets(PATH, datasetName=dsName ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.0, s=512, transformation=dict["transformation"]).getDataLoaders()
    plotName = f'Model-{dir}_DS-{dsName}_sample.png'
    os.listdir(os.path.join(PATH, "Dict", dir))[0]
    dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
    diffSolv = DiffSur().to(device)
    try:
            ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
            myLog.logging.info(f'Using Best model!')
    except:
        ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
    theModel.eval();
    plotSampRelative(theModel, testloader, dict, device, PATH, plotName, maxvalue=0.5, power=2.0)
    myLog.logging.info(f'Sample generated')
##################  
    
# dsName = "19SourcesRdm"
#     os.listdir(os.path.join(PATH, "Dict", dir))[0]
#     dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#     diffSolv = DiffSur().to(device)
#     ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
#     theModel.eval();
#     try:
# #         print(dict["transformation"])
#         trainloader, testloader = generateDatasets(PATH, datasetName=dsName, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
#     except:
#         trainloader, testloader = generateDatasets(PATH, datasetName=dsName, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512).getDataLoaders()

#     xi, yi, zi = predVsTarget(testloader, theModel, device, transformation = dict["transformation"], threshold = 0.0, nbins = 100, BATCH_SIZE = BATCH_SIZE, size = 512, lim = 10)
#     dataname = f'Model-{dir}_DS-{dsName}.txt'
#     np.savetxt(os.path.join(PATH, "AfterPlots", dataname), zi.reshape(100,100).transpose())

#     power = 1/8
#     plotName = f'Model-{dir}_DS-{dsName}_pow-{power}.png'
#     plt.pcolormesh(xi, yi, np.power(zi.reshape(xi.shape) / zi.reshape(xi.shape).max(),1/8), shading='auto')
#     plt.plot([0,1],[0,1], c='r', lw=0.2)
#     plt.xlabel("Target")
#     plt.ylabel("Prediction")
#     plt.title(f'Model {dir},\nDataset {dsName}')
#     plt.colorbar()
#     plt.savefig(os.path.join(PATH, "AfterPlots", plotName), transparent=False)
#     plt.show()
    
# python -W ignore tests.py --dir 1DGX