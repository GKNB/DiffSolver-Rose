import csv
import os

import torch
import torch.optim as optim

import util


def save_model(save_loc, model, opt, error_list, epoch):
    file_name = os.path.join(save_loc, 'diffusion-model.pt')
    torch.save({'epoch'              : epoch,
                'model_state_dict'   : model.state_dict(),
                'optimizer_sate_dict': opt.state_dict(),
                'loss'               : error_list}, file_name)
    return


class Train:

    def __init__(self, device, std_tr, s, custom_loss=None):
        self.device = device
        self.std_tr = std_tr
        self.s = s

        if custom_loss is None:
            self.loss = self.my_loss
        else:
            self.loss = custom_loss

    @staticmethod
    def my_loss(output, target, alph=1, w=1, w2 = 2000):
#         loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/w) * torch.abs((output - target)**alph))
        loss = torch.mean((1 + torch.tanh(w*target) * w2) * torch.abs((output - target)**alph))
#         dict["w"] = w
#         dict["w2"] = w2
#         dict["alph"] = alph
        return loss
#     @staticmethod
#     def my_loss(output, target, alph=2, w=1):
#         loss = torch.mean(torch.exp(-(torch.ones_like(output) - output) / w) * torch.abs((output - target) ** alph))
#         return loss

    @staticmethod
    def test_diff_error(neural_net, loader, criterion, device):
        neural_net.eval()
        with torch.no_grad():
            error = 0.0
            for i, data in enumerate(loader):
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = neural_net(x)
                err = criterion(yhat, y)
                error += err.item()
        neural_net.train()
        return error / (i + 1)

    def train_diff_solver(self, load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, epochs=100, snap=25,
                          dataset_name='TwoSourcesRdm', transformation='linear'):
        from datetime import datetime
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        save_loc = os.path.join(save_loc, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        print('Save location:', save_loc)

        diff_solver = util.NNets.SimpleCNN().to(self.device)
        criterion = self.loss

        opt = optim.Adam(diff_solver.parameters(), lr=lr)
        train_loader, test_loader = util.loaders.generateDatasets(PATH=load_loc,
                                                                  datasetName=dataset_name,
                                                                  batch_size=BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  std_tr=self.std_tr,
                                                                  s=self.s,
                                                                  transformation=transformation).getDataLoaders()
        name_plus, name_minus = util.get_Nplus_Nminus(dataset_name)

        test_loader_plus, test_loader_minus = None, None

        if name_plus is not None:
            b_size = max(BATCH_SIZE, 1)
            _, test_loader_plus = util.loaders.generateDatasets(PATH=load_loc,
                                                                datasetName=name_plus,
                                                                batch_size=b_size,
                                                                num_workers=NUM_WORKERS,
                                                                std_tr=self.std_tr, s=self.s,
                                                                transformation=transformation).getDataLoaders()
        if name_minus is not None:
            b_size = max(BATCH_SIZE, 1)
            _, test_loader_minus = util.loaders.generateDatasets(PATH=load_loc,
                                                                 datasetName=name_minus,
                                                                 batch_size=b_size,
                                                                 num_workers=NUM_WORKERS,
                                                                 std_tr=self.std_tr, s=self.s,
                                                                 transformation=transformation).getDataLoaders()
        error_list = []
        test_error, test_error_plus, test_error_minus = [], [], []
        # acc, accTrain = [], []

        # todo: add plots back

        for epoch in range(epochs):
            error = 0.0
            for (i, data) in enumerate(train_loader):
                diff_solver.zero_grad()
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                yhat = diff_solver(x)
                err = criterion(yhat, y)
                err.backward()
                opt.step()
                error += err.item()
            error_list.append(error / (i + 1))

            test_error.append(self.test_diff_error(diff_solver, test_loader, criterion, device))
            if test_loader_plus is not None:
                test_error_plus.append(self.test_diff_error(diff_solver, test_loader_plus, criterion, device))
            if test_loader_minus is not None:
                test_error_minus.append(self.test_diff_error(diff_solver, test_loader_minus, criterion, device))

            with open(os.path.join(save_loc, 'train_error.csv'), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(error_list)

            with open(os.path.join(save_loc, 'test_error.csv'), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(test_error)

            if len(test_error_plus):
                with open(os.path.join(save_loc, 'test_N+1_error.csv'), 'w+') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_error_plus)

            if len(test_error_minus):
                with open(os.path.join(save_loc, 'test_N-1_error.csv'), 'w+') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_error_minus)
            if epoch % snap == snap - 1:
                save_model(save_loc, diff_solver, opt, error_list, epoch)
        print('TRAINING FINISHED')
        return error_list, test_error, test_error_plus, test_error_minus, save_loc