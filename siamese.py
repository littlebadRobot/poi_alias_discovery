import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from nets.siamese import Siamese as siamese
from config import input_size

class Siamese(object):
    _defaults = {
        "model_path"    : 'logs/model.pth',
        "input_shape"   : (input_size[0], input_size[1], 1),
        "cuda"          : True
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
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
        

    def detect_image(self, one_pair_batch):
        images, targets = one_pair_batch[0], one_pair_batch[1]

        with torch.no_grad():
            if self.cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor)).cuda()
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor))
                
            output = self.net(images)[0]
            output = torch.nn.Sigmoid()(output)
        
        return output
