
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from torch.utils.data import DataLoader
from nets.siamese import Siamese as siamese
from utils.dataloader_own_dataset import \
    SiameseDataset as SiameseDataset_own_dataset
from utils.dataloader_own_dataset import dataset_collate
from config import input_size

from universal_utils.universal_utils import *

class Siamese(object):
    _defaults = {
        "model_path"    : 'logs/Epoch4-Total_Loss0.3427-Val_Loss0.7452.pth',
        "input_shape"   : (input_size[0], input_size[1], 1),
        "cuda"          : torch.cuda.is_available()
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()


    def run_testing(self):
        test_pos_path = "/notebook/train_test_v2/train_test/{size}/train/pos_pair.pkl".format(size=input_size[0])
        test_neg_path = "/notebook/train_test_v2/train_test/{size}/train/neg_pair.pkl".format(size=input_size[1])
        input_shape = [input_size[0],input_size[1],1]
        batch_size = 4
        test_dataset = SiameseDataset_own_dataset(input_shape, test_pos_path, test_neg_path)
        test_gen = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)
        num_test = len(load_pickle(test_neg_path))
        epoch_size_test = num_test//batch_size
        print('Start testing')
        total_accuracy = 0
        test_total_accuracy = 0
        total_loss = 0
        test_loss = 0

        Epoch = 10
        epoch = 1
        loss = nn.BCELoss()
        
        TP, TN, FN, FP = 0, 0, 0, 0
        
        for epoch in range(0,10):      
            with tqdm(total=epoch_size_test, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
                for iteration, batch in enumerate(test_gen):
                    if iteration >= epoch_size_test:
                        break
                    images_val, targets_val = batch[0], batch[1]
                    with torch.no_grad():
                        if self.cuda:
                            images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                            targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor)).cuda()
                        else:
                            images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                            targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor))
                        outputs = nn.Sigmoid()(self.net(images_val))

                        output = loss(outputs, targets_val)

                        equal = torch.eq(torch.round(outputs),targets_val)

                        TP += ((torch.round(outputs).data == 1) & (targets_val.data == 1)).sum()
                        TN += ((torch.round(outputs).data == 0) & (targets_val.data == 0)).sum()
                        FN += ((torch.round(outputs).data == 0) & (targets_val.data == 1)).sum()
                        FP += ((torch.round(outputs).data == 1) & (targets_val.data == 0)).sum()

                    p = TP.float().item() / (TP.float().item() + FP.float().item())
                    r = TP.float().item() / (TP.float().item() + FN.float().item())
                    F1 = 2 * r * p / (r + p)
                    acc = (TP.float().item() + TN.float().item()) / (TP.float().item() + TN.float().item() + FP.float().item() + FN.float().item())

                    test_loss += output.item()

                    pbar.set_postfix(**{'total_loss': test_loss / (iteration + 1), 
                                        'precision' : p, 
                                        'recall'    : r,
                                        'F1'        : F1, 
                                        'accuracy'  : acc,
                                        'accuracy2' : acc2
                                       })
                    pbar.update(1)

        print('Finish testing')
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Saving state, iter:', str(epoch+1))


if __name__ == "__main__":
    model = Siamese()
    model.run_testing()