import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from utils.utils import *

from nets.siamese import Siamese
from utils.dataloader import SiameseDataset, dataset_collate
from utils.dataloader_own_dataset import \
    SiameseDataset as SiameseDataset_own_dataset
from utils.dataloader_own_dataset import dataset_collate
from config import input_size_local, input_size_global


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, loss, epoch_idx, epoch_size, epoch_size_val, gen, genval, Epoch_total, cuda):
    
    total_loss = 0
    val_loss = 0
    total_accuracy = 0
    val_total_accuracy = 0
    TP, TN, FN, FP = 0.000000001, 0.000000001, 0.000000001, 0.000000001
    TP_val, TN_val, FN_val, FP_val = 0.000000001, 0.000000001, 0.000000001, 0.000000001
    
    with tqdm(total=epoch_size, desc=f'Epoch {epoch_idx + 1}/{Epoch_total}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):

            if iteration >= epoch_size:
                break
            images, images_gl, targets = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    images_gl = Variable(torch.from_numpy(images_gl).type(torch.FloatTensor)).cuda()
                    targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor)).cuda()
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    images_gl = Variable(torch.from_numpy(images_gl).type(torch.FloatTensor))
                    targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor))

            optimizer.zero_grad()
            outputs = nn.Sigmoid()(net(images, images_gl))
            output = loss(outputs, targets)
            output.backward()
            optimizer.step()

            with torch.no_grad():
                equal = torch.eq(torch.round(outputs), targets)
                TP += ((torch.round(outputs).data == 1) & (targets.data == 1)).sum()
                TN += ((torch.round(outputs).data == 0) & (targets.data == 0)).sum()
                FN += ((torch.round(outputs).data == 0) & (targets.data == 1)).sum()
                FP += ((torch.round(outputs).data == 1) & (targets.data == 0)).sum()
                p = TP.float().item() / (TP.float().item() + FP.float().item())
                r = TP.float().item() / (TP.float().item() + FN.float().item())
                F1 = 2 * r * p / (r + p)
                acc = (TP.float().item() + TN.float().item()) / (TP.float().item() + TN.float().item() + FP.float().item() + FN.float().item())
                total_loss += output.item()
                
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'precision' : p, 
                                'recall'    : r,
                                'F1'        : F1, 
                                'accuracy'  : acc,
                                'lr'        : get_lr(optimizer)
                               })
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch_idx + 1}/{Epoch_total}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, images_gl_val, targets_val = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    images_gl_val = Variable(torch.from_numpy(images_gl_val).type(torch.FloatTensor)).cuda()
                    targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor)).cuda()
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    images_gl_val = Variable(torch.from_numpy(images_gl_val).type(torch.FloatTensor))
                    targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor))
                optimizer.zero_grad()
                outputs = nn.Sigmoid()(net(images_val, images_gl_val))
                output = loss(outputs, targets_val)

                equal = torch.eq(torch.round(outputs),targets_val)
                TP_val += ((torch.round(outputs).data == 1) & (targets_val.data == 1)).sum()
                TN_val += ((torch.round(outputs).data == 0) & (targets_val.data == 0)).sum()
                FN_val += ((torch.round(outputs).data == 0) & (targets_val.data == 1)).sum()
                FP_val += ((torch.round(outputs).data == 1) & (targets_val.data == 0)).sum()
                p_val = TP_val.float().item() / (TP_val.float().item() + FP_val.float().item())
                r_val = TP_val.float().item() / (TP_val.float().item() + FN_val.float().item())
                F1_val = 2 * r_val * p_val / (r_val + p_val)
                acc_val = (TP_val.float().item() + TN_val.float().item()) / (TP_val.float().item() + TN_val.float().item() + FP_val.float().item() + FN_val.float().item())


            val_loss += output.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1), 
                                'precision' : p_val, 
                                'recall'    : r_val,
                                'F1'        : F1_val, 
                                'accuracy'  : acc_val
                               })
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch_idx+1) + '/' + str(Epoch_total))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch_idx+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-INPUT_Size%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch_idx+1), input_size[0],total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":

    size_local = input_size_local[0]
    size_global = input_size_global[0]
    
    train_pos_path = "/notebook/train_test_v2/train_test/{size}/train/pos_pair.pkl".format(size=size_local)
    train_neg_path = "/notebook/train_test_v2/train_test/{size}/train/neg_pair.pkl".format(size=size_local)
    test_pos_path = "/notebook/train_test_v2/train_test/{size}/test/pos_pair.pkl".format(size=size_local)
    test_neg_path = "/notebook/train_test_v2/train_test/{size}/test/neg_pair.pkl".format(size=size_local)
    val_pos_path = "/notebook/train_test_v2/train_test/{size}/val/pos_pair.pkl".format(size=size_local)
    val_neg_path = "/notebook/train_test_v2/train_test/{size}/val/neg_pair.pkl".format(size=size_local)

    input_shape_local = [size_local,size_local,1]
    input_shape_global = [size_global, size_global, 1]
    pretrained = False
    Cuda = torch.cuda.is_available()
    
    model = Siamese(input_shape_local, input_shape_global, pretrained)

    # model_path = "model_data/Omniglot_vgg.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    
    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    loss = nn.BCELoss()

    num_train = len(load_pickle(train_neg_path))
    num_val = len(load_pickle(val_neg_path))
    
    # warm-up
    Batch_size = 16
    lr = 3e-4
    Init_Epoch = 0
    Freeze_Epoch = 10

    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

    train_dataset = SiameseDataset_own_dataset(input_shape, train_pos_path, train_neg_path)
    val_dataset = SiameseDataset_own_dataset(input_shape, val_pos_path, val_neg_path)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True, 
                            drop_last=True, collate_fn=dataset_collate)

    epoch_size = max(1, num_train//Batch_size)
    epoch_size_val = num_val//Batch_size

    for epoch_idx in range(Init_Epoch, Freeze_Epoch):
        fit_one_epoch(net, loss, epoch_idx, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
        lr_scheduler.step()

    
    Batch_size = 32
    lr = 1e-4
    Freeze_Epoch = 10
    Unfreeze_Epoch = 50

    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

    train_dataset = SiameseDataset_own_dataset(input_shape, dataset_path, num_train, num_val, train=True)
    val_dataset = SiameseDataset_own_dataset(input_shape, dataset_path, num_train, num_val, train=False)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                            drop_last=True, collate_fn=dataset_collate)

    epoch_size = max(1, num_train//Batch_size)
    epoch_size_val = num_val//Batch_size

    for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
        fit_one_epoch(net,loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
        lr_scheduler.step()
