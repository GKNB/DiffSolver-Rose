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


sys.path.insert(1, '/home/javier/Projects/DiffSolver/DeepDiffusionSolver/util')

from loaders import generateDatasets, inOut, saveJSON, loadJSON#, MyData
from NNets import SimpleCNN
from tools import accuracy, tools, per_image_error, predVsTarget, errInDS, numOfPixels
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Deep Diffusion Solver")
    parser.add_argument('--path', dest="path", type=str, default="/raid/javier/Datasets/DiffSolver/",
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

    selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []} }}

    datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
    error, errorField, errorSrc = [], [], []

    for selectedDir in selectedDirs.keys():
        dir = selectedDir
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#        print(dict, '\n')
#         myLog = thelogger()
#         myLog.logFunc(PATH, dict, dir)
#         myLog.logging.info(f'Generating tests using MSE... for model {selectedDir}')
#         try:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict, tag='Best')
#             myLog.logging.info(f'Using Best model!')
#         except:
#             ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
#         theModel.eval();
        
        for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
    
            arr = numOfPixels(testloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
            selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
            selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
            selectedDirs[selectedDir]["mean"]["src"].append(arr[2])

            selectedDirs[selectedDir]["mean"]["ring1"].append(arr[3])
            selectedDirs[selectedDir]["mean"]["ring2"].append(arr[4])
            selectedDirs[selectedDir]["mean"]["ring3"].append(arr[5])
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
    saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots"), f'stats_test.json')
#     myLog.logging.info(f'JSON object saved')

    selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []} }}

    for selectedDir in selectedDirs.keys():
        dir = selectedDir
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
        
        for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
    #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=512)
            arr = numOfPixels(trainloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
            selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
            selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
            selectedDirs[selectedDir]["mean"]["src"].append(arr[2])

            selectedDirs[selectedDir]["mean"]["ring1"].append(arr[3])
            selectedDirs[selectedDir]["mean"]["ring2"].append(arr[4])
            selectedDirs[selectedDir]["mean"]["ring3"].append(arr[5])
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
    saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots"), f'stats_train.json')
#     myLog.logging.info(f'JSON object saved')
    

    
# python -W ignore trainSetStats.py --dir 1DGX