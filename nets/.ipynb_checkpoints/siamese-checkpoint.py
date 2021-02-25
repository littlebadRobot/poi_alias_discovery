import torch
import torch.nn as nn
from nets.backbone import local_net, global_net


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        block_num = 3
        filter_sizes = [2]*block_num
        padding = [0]*block_num
        stride = 2
        for i in range(len(filter_sizes)):
            input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width)*get_output_length(height) 

def get_img_output_length_gl(width, height):
    def get_output_length(input_length):
        # input_length += 6
        block_num = 4
        filter_sizes = [2]*block_num
        padding = [0]*block_num
        stride = 2
        for i in range(1):
            input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width)*get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, input_shape_gl,  pretrained=False):
        super(Siamese, self).__init__()
        self.net_local = local_net(pretrained, input_shape[-1])
        del self.net_local.avgpool
        del self.net_local.classifier
        
        self.net_global = global_net(pretrained, input_shape_gl[-1])
        del self.net_global.avgpool
        del self.net_global.classifier
        
        flat_shape_local = 512 * get_img_output_length(input_shape[1], input_shape[0])
        flat_shape_global = 512 * get_img_output_length_gl(input_shape_gl[1], input_shape_gl[0]) 
        flat_shape_cat = flat_shape_local + flat_shape_local

        self.fully_connect1 = torch.nn.Linear(flat_shape_cat, 512)
        self.fully_connect2 = torch.nn.Linear(512,1)
        

    def forward(self, x, x_gl):
        
        x1_gl, x2_gl = x_gl
        x1, x2 = x

        x1 = self.net_local.features(x1)
        x2 = self.net_local.features(x2) 
        x1_gl = self.net_global.features(x1_gl)
        x2_gl = self.net_global.features(x2_gl) 
    
        x1 = torch.flatten(x1,1)
        x2 = torch.flatten(x2,1)
        x1_gl = torch.flatten(x1_gl,1)
        x2_gl = torch.flatten(x2_gl,1)
        
        A=torch.cat((x1,x1_gl),1)
        B=torch.cat((x2,x2_gl),1)
        
        A = self.fully_connect1(A)
        B = self.fully_connect1(B)
        x = torch.abs(A - B)
        x = self.fully_connect2(x)
        
        return x