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
from NNets import SimpleCNN, DiffDisc, SimpleCNN_10K, SimpleCNN_5K
from tools import accuracy, tools, per_image_error, predVsTarget
from plotter import myPlots, plotSamp, plotSampRelative

class DiffSur(SimpleCNN):
    pass

class DiffDiscriminator(DiffDisc):
    pass

def inverse_huber_loss(target,output, C=0.5):
    absdiff = torch.abs(output-target)
#     C = 0.5#*torch.max(absdiff).item()
#     return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
    return torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) )

@torch.no_grad()
def rankScores(dsTemp="AllHalf"):
    myLog = inOut()
    myLog.logFunc(PATH, dict, dir)
    myLog.logging.info('Ranking Dataset')
    
    diffSolv = DiffSur().to(device)
    lastEpoch, _, diffSolv = inOut().load_model(diffSolv, "Diff", dict)
    loss = lambda  output, target : torch.mean( torch.abs((output - target)**1.0))
    dsObj = generateDatasets(PATH, datasetName=dsTemp, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, s=512, transformation="linear")
    scores = [0 for i in range(dsObj.df_train.shape[0])]
    for i in range(dsObj.df_train.shape[0]):
        r = dsObj.train.__getitem__(i)
        y1 = diffSolv(r[0].unsqueeze(0).to(device))
        scores[i] = loss(y1, r[1].to(device)).item()
        myLog.logging.info(f'{i}/{dsObj.df_train.shape[0]}')
    sortDF = dsObj.df_train.iloc[list(np.argsort(scores)[::-1][:2000])].copy()
    allsubDF = pd.read_csv(os.path.join(PATH) + "/trainSubTemp.csv")
    newDF = pd.concat([sortDF, allsubDF.iloc[:2000]]).reset_index()
    newDF.to_csv(os.path.join(PATH) + "/trainSubTemp.csv")
    myLog.logging.info('New temp CSVs created')
    myLog.logging.close()

class train(object):
    def __init__(self, latentSpace=100, std_tr=0.0, s=512, transformation="linear", wtan=10, w=1, w2=6000, select="step", alph=1, delta=0.02, toggleIdx=1, p=0.5):
        self.real_label = 1.0
        self.fake_label = 0.0
        self.latentSpace = latentSpace
        self.std_tr = std_tr
        self.s = s
        self.trans = transformation
        dict["transformation"] = self.trans
        dict["w"] = w
        dict["wtan"] = wtan
        dict["w2"] = w2
        dict["alph"] = alph
        dict["lossSelection"] = select
        dict["delta"] = delta
        dict["togIdx"] = toggleIdx
        dict["p"] = p
        
    def my_loss(self, output, target, ep, dict):
        if dict["lossSelection"] == "step":
            loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) *dict["w2"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "exp":
            loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "huber":
            loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) * dict["w2"]) * torch.nn.HuberLoss(reduction='none', delta=dict["delta"])(output, target))
        elif dict["lossSelection"] == "toggle":
            if np.mod(np.divmod(ep, dict["togIdx"])[0], 2) == 0:
                loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) * dict["w2"]) * torch.abs((output - target)**dict["alph"]))
            else:
                loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "rand":
#             r = np.random.rand()            
            if dict["r"][-1]<dict["p"]:
                loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) * dict["w2"]) * torch.abs((output - target)**dict["alph"]))
            else:
                loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "invhuber":
            loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * inverse_huber_loss(target,output, C = dict["delta"]))
        elif dict["lossSelection"] == "invhuber2":
            loss = torch.mean(inverse_huber_loss(target,output, C = dict["delta"]))
        return loss

    def trainClass(self, epochs=100, snap=25):
        clas = MLP().to(device)
        criterion = nn.NLLLoss()
        # criterion = my_loss
        opt = optim.Adam(clas.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list = []
        acc, accTrain = [], []
        for epoch in range(epochs):
            error = 0.0
            for (i, data) in enumerate(trainloader):
                clas.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = clas(x)
                err = criterion(yhat,y)
                err.backward()
                opt.step()
                error += err.item()
                # if i > 2:
                #     break
            error_list.append(error/(i+1))
            acc.append(accuracy().validation(testloader, clas).item())
            accTrain.append(accuracy().validation(trainloader, clas).item())
            # acc.append(i)
            # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            # ax1.plot(error_list, 'b*-', lw=3, ms=12)
            # ax1.set(ylabel='loss', title='Epoch {}'.format(epoch+1))
            # ax2.plot(acc, 'r*-', lw=3, ms=12)
            # ax2.plot(accTrain, 'g*-', lw=3, ms=12)
            # ax2.set(xlabel='epochs', ylabel='%', title="Accuracy")
            # plt.show()
            myPlots().clasPlots(error_list, acc, accTrain, epoch)

            if epoch % snap == snap-1 :
                inOut().save_model(clas, 'Class', opt, error_list, epoch, dir)
        print("Done!!")
        return error_list, acc, accTrain

    def trainDiffSolver(self, epochs=100, snap=25, bashTest=False):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Training Diff Solver')
        diffSolv = DiffSur().to(device)
        # criterion = nn.NLLLoss()
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = [], []
        acc, accTrain = [], []
        dict["r"] = []
#         myPlots().plotDiff(PATH, dir, device, [1,2,3], [2,3,4], testloader, diffSolv, 1, transformation=self.trans, bash=True)
        for epoch in range(epochs):
            dict["r"].append( np.random.rand())
            error = 0.0
            for (i, data) in enumerate(trainloader):
                diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
                err.backward()
                opt.step()
                # if i > 2:
                #     break
#                 print(f'Epoch: {epoch}/{epochs}. Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            myPlots().plotDiff(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) #<----------------------------------------------
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")
        return error_list#, acc, accTrain
    
    def trainDiffWGANs(self, epochs=100, snap=25, bashTest=False):
        torch.autograd.set_detect_anomaly(True)
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Training Diff Solver w/ WGANs')
        diffSolv = DiffSur().to(device)
        disc = DiffDiscriminator().to(device)
        
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = [], []
        acc, accTrain = [], []
        dict["r"] = []
        
        wass = lambda x : x.mean()
        criterionD = wass
        #OPT
        optD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.99))
#         optD = optim.RMSprop(disc.parameters(), lr=self.dict['lr'])

        error_list_D = []
        error_list_G = []

        for epoch in range(epochs):
            dict["r"].append( np.random.rand())
            error = 0.0
            errorD = 0.0
            errorG = 0.0
            for (i, data) in enumerate(trainloader):
#                 diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)

                disc.zero_grad()
                yreal = disc(y)
                yfake = disc(yhat.detach())
                errD = - criterionD(yreal) + criterionD(yfake)
                errD.backward() # yfake = disc(yhat.detach())
#                 torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.01)
                optD.step()
                errorD += errD.item()

                diffSolv.zero_grad()
                yfake = disc(yhat)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
#                 err.backward(retain_graph=True)
#                 opt.step()
                errG = - criterionD(yfake) + err 
                errG.backward()
                opt.step()
                errorG += errG.item()

                # if i > 2:
                #     break
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}, errorD: {errorD/(i+1)}, erroG: {errorG/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            error_list_D.append(errorD/(i+1))
            error_list_G.append(errorG/(i+1))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            myPlots().plotDiff(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) 
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, disc, 'Disc', optD, error_list, error_list_test, epoch, dir)
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")

        return error_list#, acc, accTrain
    
    def trainDiffGANs(self, epochs=100, snap=25, bashTest=False):
#         torch.autograd.set_detect_anomaly(True)
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Training Diff Solver w/ GANs')
        diffSolv = DiffSur().to(device)
        disc = DiffDiscriminator().to(device)
        
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = [], []
        acc, accTrain = [], []
        dict["r"] = []
        
#         wass = lambda x : x.mean()
        criterionD = nn.BCELoss()
        #OPT
        optD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.99))
#         optD = optim.RMSprop(disc.parameters(), lr=self.dict['lr'])

        error_list_D = []
        error_list_G = []

        for epoch in range(epochs):
            dict["r"].append( np.random.rand())
            error = 0.0
            errorD = 0.0
            errorG = 0.0
            for (i, data) in enumerate(trainloader):
#                 diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)

                disc.zero_grad()
                ycatR = torch.cat([y, x], 1)
                yreal = disc(ycatR)
                yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
                ycatF = torch.cat([yhat, x], 1)
                yfake = disc(ycatF.detach())
                yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
                errD = dict["GAN_reg"] * (criterionD(yreal,yR) + criterionD(yfake,yF))
                errD.backward() 
#                 torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.01)
                optD.step()
                errorD += errD.item()

                diffSolv.zero_grad()
                yfake = disc(ycatF)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
#                 err.backward(retain_graph=True)
#                 opt.step()
                errG = dict["GAN_reg"] * criterionD(yfake, yR) + err 
                errG.backward()
                opt.step()
                errorG += errG.item()

                # if i > 2:
                #     break
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}, errorD: {errorD/(i+1)}, erroG: {errorG/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            error_list_D.append(errorD/(i+1))
            error_list_G.append(errorG/(i+1))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            myPlots().plotDiff(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) 
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, disc, 'Disc', optD, error_list, error_list_test, epoch, dir)
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")

        return error_list#, acc, accTrain


    def continuetrainDiffSolver(self, epochs=100, snap=25):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Continue Training Diff Solver')
        diffSolv = DiffSur().to(device)
        # criterion = nn.NLLLoss()
        # load_model(self, module, dict)
        lastEpoch, _, diffSolv = inOut().load_model(diffSolv, "Diff", dict)
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        myLog.logging.info(f'Last Epoch = {lastEpoch}')
        error_list, error_list_test = dict["Loss"], dict["LossTest"]
        acc, accTrain = [], []
        for epoch in range(lastEpoch+1, epochs):
            dict["r"].append( np.random.rand())
            error = 0.0
            for (i, data) in enumerate(trainloader):
                diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
                err.backward()
                opt.step()
                # if i > 2:
                #     break
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            myPlots().plotDiff(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True)

            if epoch % snap == snap-1 :
#                 inOut().save_model(PATH, dict, disc, 'Disc', optD, error_list, error_list_test, epoch, dir)
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
        myLog.logging.info("Done!!")
        return error_list#, acc, accTrain
    
    def continuetrainDiffWGANs(self, epochs=100, snap=25, bashTest=False):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Continue Training Diff Solver w/ WGANs')
        diffSolv = DiffSur().to(device)
        lastEpoch, _, diffSolv = inOut().load_model(diffSolv, "Diff", dict)
        disc = DiffDiscriminator().to(device)
        _, _, disc = inOut().load_model(diffSolv, "Disc", dict)
        
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = dict["Loss"], dict["LossTest"]
        acc, accTrain = [], []
        
        wass = lambda x : x.mean()
        criterionD = wass
        #OPT
        optD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.99))
#         optD = optim.RMSprop(disc.parameters(), lr=self.dict['lr'])

        error_list_D = []
        error_list_G = []

        for epoch in range(lastEpoch+1, epochs):
            dict["r"].append( np.random.rand())
            error = 0.0
            errorD = 0.0
            errorG = 0.0
            for (i, data) in enumerate(trainloader):
#                 diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)

                disc.zero_grad()
                yreal = disc(y)
                yfake = disc(yhat.detach())
                errD = - criterionD(yreal) + criterionD(yfake)
                errD.backward() # yfake = disc(yhat.detach())
#                 torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.01)
                optD.step()
                errorD += errD.item()

                diffSolv.zero_grad()
                yfake = disc(yhat)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
#                 err.backward(retain_graph=True)
#                 opt.step()
                errG = - criterionD(yfake) + err 
                errG.backward()
                opt.step()
                errorG += errG.item()

                # if i > 2:
                #     break
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}, errorD: {errorD/(i+1)}, erroG: {errorG/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            error_list_D.append(errorD/(i+1))
            error_list_G.append(errorG/(i+1))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            myPlots().plotDiff(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) 
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, disc, 'Disc', optD, error_list, error_list_test, epoch, dir)
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")

        return error_list#, acc, accTrain
    
    def continuetrainDiffGANs(self, epochs=100, snap=25, bashTest=False):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Continue Training Diff Solver w/ GANs')
        diffSolv = DiffSur().to(device)
        lastEpoch, _, diffSolv = inOut().load_model(diffSolv, "Diff", dict)
        disc = DiffDiscriminator().to(device)
        _, _, disc = inOut().load_model(disc, "Disc", dict)
        
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = dict["Loss"], dict["LossTest"]
        acc, accTrain = [], []
        
        criterionD = nn.BCELoss()
        #OPT
        optD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.99))
#         optD = optim.RMSprop(disc.parameters(), lr=self.dict['lr'])

        error_list_D = []
        error_list_G = []

        for epoch in range(lastEpoch+1, epochs):
            dict["r"].append( np.random.rand())
            error = 0.0
            errorD = 0.0
            errorG = 0.0
            for (i, data) in enumerate(trainloader):
#                 diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)

                disc.zero_grad()
                yreal = disc(y)
                yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
                yfake = disc(yhat.detach())
                yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
                errD = dict["GAN_reg"] * (criterionD(yreal,yR) + criterionD(yfake,yF))
                errD.backward() # yfake = disc(yhat.detach())
#                 torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.01)
                optD.step()
                errorD += errD.item()

                diffSolv.zero_grad()
                yfake = disc(yhat)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
#                 err.backward(retain_graph=True)
#                 opt.step()
                errG = dict["GAN_reg"] * criterionD(yfake, yR) + err
                errG.backward()
                opt.step()
                errorG += errG.item()

                # if i > 2:
                #     break
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}, errorD: {errorD/(i+1)}, erroG: {errorG/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            error_list_D.append(errorD/(i+1))
            error_list_G.append(errorG/(i+1))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            myPlots().plotDiff(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) 
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, disc, 'Disc', optD, error_list, error_list_test, epoch, dir)
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")

        return error_list#, acc, accTrain

    def computeError(self, testloader, theModel, ep, dict):  
        criterion = self.my_loss
        with torch.no_grad():
            erSum = 0
            for (i, data) in enumerate(testloader):
                x = data[0].to(device)
                y = data[1].to(device)
#                 erSum += torch.mean(torch.abs(theModel(x) - y)).item()
                erSum += criterion(theModel(x), y, ep, dict).item()

            return erSum / len(testloader)

    def saveBestModel(self, PATH, dict, theModel, module, opt, error_list, error_list_test, epoch, dir, tag='Best'):
        if error_list_test.index(np.min(error_list_test)) == len(error_list_test)-1:
            inOut().save_model(PATH, dict, theModel, module, opt, error_list, error_list_test, epoch, dir, tag=tag)
            #save model
            

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
    parser.add_argument('--p', dest="p", type=float, default=1,
                        help="Specify hparam p")
    parser.add_argument('--disc', dest="disc", type=bool, default=False,
                        help="Using discriminator?")
    parser.add_argument('--ganr', dest="ganr", type=float, default=1.0,
                        help="Specify hparam GAN regularizer")
    parser.add_argument('--seed', dest="seed", type=bool, default=False,
                        help="Set seed to 0?")
      
    args = parser.parse_args()
    
    if args.seed:
        torch.manual_seed(0)
        np.random.seed(0)
    
    
    ###Start Here
    PATH = args.path # "/raid/javier/Datasets/DiffSolver/"
    DATASETNAME = args.dataset # "All"

    dir = args.dir #'1DGX' #'Test'#str(21)
    BATCH_SIZE=args.bs #50
    NUM_WORKERS=args.nw #8
    ngpu = args.ngpu #1
    lr = args.lr #0.0001
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if args.newdir:
        dict = inOut().newDict(PATH, dir)
    else:
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
    dict["lr"]=lr
    
    
    if "dataset" in dict:
        dict["dataset"].append(DATASETNAME)
    else:
        dict["dataset"] = [DATASETNAME]


    if args.disc == False:
        if args.newtrain:
            dict["seed"] = args.seed
            error_list = train(latentSpace=100, std_tr=0.0, s=512, transformation=args.transformation, wtan=args.wtan, w=args.w, alph=args.alpha, w2=args.w2, select=args.loss, delta=args.delta, toggleIdx=args.toggle, p=args.p).trainDiffSolver(args.maxep,1, bashTest=args.bashtest)
        else:
            if DATASETNAME == "AllSubTemp":
#                 rankScores()
                error_list = train(latentSpace=100, std_tr=0.0, s=512, transformation=dict["transformation"], wtan=dict["wtan"], w=dict["w"], alph=dict["alph"], w2=dict["w2"], select=dict["lossSelection"], delta=dict["delta"], toggleIdx=dict["togIdx"], p=dict["p"]).continuetrainDiffSolver(args.maxep,1)
            else:
                error_list = train(latentSpace=100, std_tr=0.0, s=512, transformation=dict["transformation"], wtan=dict["wtan"], w=dict["w"], alph=dict["alph"], w2=dict["w2"], select=dict["lossSelection"], delta=dict["delta"], toggleIdx=dict["togIdx"], p=dict["p"]).continuetrainDiffSolver(args.maxep,1)
    else:
        if args.newtrain:
            dict["GAN_reg"] = args.ganr
            error_list = train(latentSpace=100, std_tr=0.0, s=512, transformation=args.transformation, wtan=args.wtan, w=args.w, alph=args.alpha, w2=args.w2, select=args.loss, delta=args.delta, toggleIdx=args.toggle, p=args.p).trainDiffGANs(args.maxep,1, bashTest=args.bashtest)
        else:
            error_list = train(latentSpace=100, std_tr=0.0, s=512, transformation=dict["transformation"], wtan=dict["wtan"], w=dict["w"], alph=dict["alph"], w2=dict["w2"], select=dict["lossSelection"], delta=dict["delta"], toggleIdx=dict["togIdx"], p=dict["p"]).continuetrainDiffGANs(args.maxep,1)
  

   
# python trainModel.py --dataset All --dir 2DGX --newdir True --newtrain True --transformation linear --loss huber

