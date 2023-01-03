import json
import logging
import os
from datetime import datetime

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

datasetDict = {"All" : ["/test.csv", "/train.csv"], 
               "AllSubnew" : ["/testSubNEW.csv", "/trainSubNEW.csv"], 
               "AllSub" : ["/testSub.csv", "/trainSub.csv"], 
               "AllSub2" : ["/testSub-2.csv", "/trainSub-2.csv"], 
               "AllSub3" : ["/testSub-3.csv", "/trainSub-3.csv"], 
               "AllSub4" : ["/testSub-4.csv", "/trainSub-4.csv"], 
               "AllSub5" : ["/testSub-5.csv", "/trainSub-5.csv"], 
               "AllSubTemp" : ["/testSub.csv", "/trainSubTemp.csv"], 
               "AllHalf" : ["/testHalf.csv", "/trainHalf.csv"], 
               "AllDouble" : ["/test2:1.csv", "/train2:1.csv"], 
               "AllFourth" : ["/test4:1.csv", "/train4:1.csv"],
               "All1:5" : ["/test1:5.csv", "/train1:5.csv"],
               "All1:5-2" : ["/test1:5-2.csv", "/train1:5-2.csv"],
               "All1:5-3" : ["/test1:5-3.csv", "/train1:5-3.csv"],
               "All1:5-4" : ["/test1:5-4.csv", "/train1:5-4.csv"],
               "All1:5-5" : ["/test1:5-5.csv", "/train1:5-5.csv"],
               "All1:10" : ["/test1:10.csv", "/train1:10.csv"]}


class MyData(Dataset):
    def __init__(self, PATH, df, datasetName="5SourcesRdm", dset="test", std=0.25, s=512, transformation="linear",
                 constant_factor=1, power=0.25):
        self.s = s
        self.df = df
        self.datasetName = datasetName
        self.transformation = transformation
        self.constant_factor = constant_factor
        self.power = power
        if datasetName in datasetDict.keys():
            self.path = os.path.join(PATH, dset)
#         if datasetName == "All":
#             self.path = os.path.join(PATH, dset)
#         elif datasetName == "AllSub":
#             self.path = os.path.join(PATH, dset)
#         elif datasetName == "AllHalf":
#             self.path = os.path.join(PATH, dset)
        else:
            self.path = os.path.join(PATH, datasetName, dset)
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(s, interpolation=PIL.Image.NEAREST),
            AddGaussianNoise(0., std),
        ])
        self.t_noNoise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(s, interpolation=PIL.Image.NEAREST),
        ])
        # self.fileNames = os.listdir(self.path)
        # self.fileNames.sort()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
#         if self.datasetName != "All" and self.datasetName != "AllSub" and self.datasetName != "AllHalf":
        if self.datasetName not in datasetDict.keys():
            file = self.path + "/" + self.df.Cell[index]
            file2 = self.path + "/" + self.df.Field[index]
        else:
            file = self.path.split("/Results/")[0] + "/Results/" + self.df.Prefix[index] + "/" + \
                   self.path.split("/Results/")[1] + "/" + self.df.Cell[index]
            file2 = self.path.split("/Results/")[0] + "/Results/" + self.df.Prefix[index] + "/" + \
                    self.path.split("/Results/")[1] + "/" + self.df.Field[index]
        image = self.load_image(file, add_noise=False)
        label = self.load_image(file2, add_noise=False)
        return image, label

    def load_image(self, file_name, add_noise=False):
        #         if self.transformation == "linear":
        #             x = np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(self.s, self.s))
        #         elif self.transformation == "sqrt":
        #             x = np.sqrt(np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(self.s, self.s)))

        #         elif self.transformation == "log":
        #             x = np.log10(
        #                 self.constant_factor + np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(self.s, self.s)))
        #         elif self.transformation == "pow":
        #             x = np.power(np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(self.s, self.s)), self.power)
        #         else:
        #             x = np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(self.s, self.s))

        x = transformation(file_name, self.transformation, self.s, power=self.power,
                           constant_factor=self.constant_factor)

        if add_noise:
            return self.t(x)
        return self.t_noNoise(x)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class generateDatasets(object):
    def __init__(self, PATH, datasetName="TwoSourcesRdm", batch_size=40, num_workers=8, std_tr=0.0, s=512,
                 transformation="linear"):
        self.bsize = batch_size
        self.nworkers = num_workers
#         if datasetName != "All" and datasetName != "AllSub" and datasetName != "AllHalf":
#             self.df_test = pd.read_csv(os.path.join(PATH, datasetName) + "/test.csv")
#             self.df_train = pd.read_csv(os.path.join(PATH, datasetName) + "/train.csv")
#         elif datasetName == "AllSub":
#             self.df_test = pd.read_csv(os.path.join(PATH) + "/testSub.csv")
#             self.df_train = pd.read_csv(os.path.join(PATH) + "/trainSub.csv")
#         elif datasetName == "AllHalf":
#             self.df_test = pd.read_csv(os.path.join(PATH) + "/testHalf.csv")
#             self.df_train = pd.read_csv(os.path.join(PATH) + "/trainHalf.csv")
#         else:
#             self.df_test = pd.read_csv(os.path.join(PATH) + "/test.csv")
#             self.df_train = pd.read_csv(os.path.join(PATH) + "/train.csv")
        
        if datasetName in datasetDict.keys():
            self.df_test = pd.read_csv(os.path.join(PATH) + datasetDict[datasetName][0])
            self.df_train = pd.read_csv(os.path.join(PATH) + datasetDict[datasetName][1])
        else:
            self.df_test = pd.read_csv(os.path.join(PATH, datasetName) + "/test.csv")
            self.df_train = pd.read_csv(os.path.join(PATH, datasetName) + "/train.csv")
        self.test = MyData(PATH, self.df_test, datasetName=datasetName, dset="test", std=0.0, s=s,
                           transformation=transformation)
        self.train = MyData(PATH, self.df_train, datasetName=datasetName, dset="train", std=std_tr, s=s,
                            transformation=transformation)

    def outputDatasets(self, typeSet="test"):
        if typeSet == "test":
            return self.test, self.df_test
        elif typeSet == "train":
            return self.train, self.df_train

    def getWeights(self):
        wTest = np.zeros(self.df_test.Label.unique().size)
        for i in range(self.df_test.Label.size):
            wTest[int(self.df_test.Label[i]) - 1] += 1
        wTrain = np.zeros(self.df_train.Label.unique().size)
        for i in range(self.df_train.Label.size):
            wTrain[int(self.df_train.Label[i]) - 1] += 1
        if np.prod(wTest == self.df_test.Label.size / len(wTest)):
            print("Labels are balanced in test set")
        if np.prod(wTrain == self.df_train.Label.size / len(wTrain)):
            print("Labels are balanced in train set")
        return wTest, wTrain

    def getDataLoaders(self):
        trainloader = torch.utils.data.DataLoader(self.train, batch_size=self.bsize,
                                                  shuffle=True, num_workers=self.nworkers)
        testloader = torch.utils.data.DataLoader(self.test, batch_size=self.bsize,
                                                 shuffle=True, num_workers=self.nworkers)
        return trainloader, testloader

    def spinVsTemp(self):
        meanSpin_Te = np.zeros(10)
        Temp_Te = np.zeros(10)
        meanSpin_Tr = np.zeros(10)
        Temp_Tr = np.zeros(10)
        jj = 10
        ii_Te = int(self.test.__len__() / jj)
        ii_Tr = int(self.train.__len__() / jj)
        for j in range(jj):
            ms = 0
            mT = 0
            for i in range(ii_Te):
                ms += self.test[j * ii_Te + i][0].mean().item()
                mT += self.test[j * ii_Te + i][1].item()
            meanSpin_Te[j] = ms / ii_Te
            Temp_Te[j] = mT / ii_Te

            ms = 0
            mT = 0
            for i in range(ii_Tr):
                ms += self.train[j * ii_Tr + i][0].mean().item()
                mT += self.train[j * ii_Tr + i][1].item()
            meanSpin_Tr[j] = ms / ii_Tr
            Temp_Tr[j] = mT / ii_Tr

        T_Te = Temp_Te / 10 + 1.8
        T_Tr = Temp_Tr / 10 + 1.8
        plt.plot(T_Te, meanSpin_Te)
        plt.legend("Test")
        plt.xlabel("Temp")
        plt.ylabel("Mean Spin")
        plt.plot(T_Tr, meanSpin_Tr)
        plt.legend("Training")
        plt.show()


class inOut(object):
    def save_model(self, PATH, dict, model, module, optimizer, loss, loss_test, epoch, dir, tag='backup'):
        # global PATH
        os.path.isdir(PATH + "Models/") or os.mkdir(PATH + "Models/")
        # date = datetime.now().__str__()
        # date = date[:16].replace(':', '-').replace(' ', '-')
        path = PATH + "Models/" + dir + "/"
        os.path.isdir(path) or os.mkdir(path)
        filename = os.path.join(path, f'{module}-{tag}.pt')
        torch.save({
            'epoch'               : epoch,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss'                : loss,
        }, filename)
        if tag == 'backup':
            if len(dict[module]) == 0: # or tag != 'backup'
                dict[module].append(filename)
            else:
                dict[module][-1] = filename
        elif tag == "Best":
            if len(dict[module+"-"+tag]) == 0: # or tag != 'backup'
                dict[module+"-"+tag].append(filename)
            else:
                dict[module+"-"+tag][-1] = filename
        dict["Loss"] = loss
        dict["LossTest"] = loss_test
        self.saveDict(dict)

    def load_model(self, net, module, dict, tag='backup'):
        if tag == 'backup':
            if module == "Class":
                model = MLP().to(device)
                checkpoint = torch.load(dict[module][-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                return last_epoch, loss, model
            elif module == "Diff":
                model = net  # DiffSur().to(device)
                checkpoint = torch.load(dict[module][-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                return last_epoch, loss, model
            elif module == "Gen":
                model = Gen().to(device)
                checkpoint = torch.load(dict[module][-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                return last_epoch, loss, model
            elif module == "Disc":
                model = net #Disc().to(device)
                checkpoint = torch.load(dict[module][-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                return last_epoch, loss, model
            elif module == "Enc":
                model = Enc().to(device)
                checkpoint = torch.load(dict[module][-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                return last_epoch, loss, model
        elif tag == 'Best':
            if module == "Diff":
                model = net  # DiffSur().to(device)
                checkpoint = torch.load(dict[module+'-'+tag][-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                return last_epoch, loss, model

    def newDict(self, PATH, dir="0"):
        # os.path.isdir(PATH) or os.mkdir(PATH)
        os.path.isdir(PATH + "Dict/") or os.mkdir(PATH + "Dict/")
        path = PATH + "Dict/" + dir + "/"
        os.path.isdir(path) or os.mkdir(path)
        date = datetime.now().__str__()
        date = date[:16].replace(':', '-').replace(' ', '-')
        DICT_NAME = f'Dict-{date}.json'
        dict = {
            "Class"   : [],
            "Gen"     : [],
            "Disc"    : [],
            "Enc"     : [],
            "Diff"    : [],
            "Diff-Best"    : [],
            "Path"    : [path + DICT_NAME],
            "Loss"    : [],
            "LossTest": [],
        }
        self.saveDict(dict)
        return dict

    def loadDict(self, dir):
        f = open(dir, "r")
        dict = json.load(f)
        f.close()
        return dict

    def saveDict(self, dict):
        dict["LastUpdated"] = [self.timestamp()]
        dictJSON = json.dumps(dict)
        f = open(dict["Path"][-1], "w")
        f.write(dictJSON)
        f.close()
        
    def timestamp(self):
        date = datetime.now().__str__()
        date = date[:19].replace(' ', '-')
        return date

    def logFunc(self, PATH, dict, dir="0"):
        self.initTime = datetime.now()
        os.path.isdir(PATH + "Logs/") or os.mkdir(PATH + "Logs/")
        os.path.isdir(PATH + "Logs/" + dir) or os.mkdir(PATH + "Logs/" + dir)
        path = PATH + "Logs/" + dir + "/"

        self.logging = logging
        self.logging = logging.getLogger()
        self.logging.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler(os.path.join(path, 'DiffSolver.log'))
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

        dict["Log"] = os.path.join(path, 'DiffSolver.log')


#         self.logging.info(f'{str(self.initTime).split(".")[0]} - Bulk Export started')
#         self.logging.info(f'bulkImport - Data will be dumped in {self.pathToDump}')

def saveJSON(obj, PATH, filename):
    dictJSON = json.dumps(obj)
    os.path.isdir(PATH) or os.mkdir(PATH)
    f = open(os.path.join(PATH, filename), "w")
    f.write(dictJSON)
    f.close()


def loadJSON(PATH, filename):
    with open(os.path.join(PATH, filename), 'r') as h:
        js = json.load(h)
    return js


@torch.no_grad()
def transformation(file_name, transformation, size, power=2, constant_factor=1):
    if transformation == "linear":
        x = np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(size, size))
    elif transformation == "sqrt":
        x = np.sqrt(np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(size, size)))

    elif transformation == "log":
        x = np.log10(
            constant_factor + np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(size, size)))
    elif transformation == "pow":
        x = np.power(np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(size, size)), power)
    else:
        x = np.absolute(np.loadtxt(file_name).astype(np.float32).reshape(size, size))

    return x


@torch.no_grad()
def transformation_inverse(yhat, y, transformation, power=2, constant_factor=1):
    if transformation == "linear":
        pass
    elif transformation == "sqrt":
        yhat = yhat.pow(2)
        y = y.pow(2)
    elif transformation == "log":
        yhat = torch.pow(torch.Tensor([10]), yhat) - torch.ones_like(yhat)
        y = torch.pow(torch.Tensor([10]), y) - torch.ones_like(y)
    elif transformation == "pow":
        yhat = yhat.pow(1.0 / power)
        y = y.pow(1.0 / power)

    return yhat, y
