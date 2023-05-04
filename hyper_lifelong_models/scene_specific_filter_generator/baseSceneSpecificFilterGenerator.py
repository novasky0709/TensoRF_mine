import torch
import numpy as np
import torch.nn as nn
class Embedding(nn.Module):
    def __init__(self,num,dim,device):
        super(Embedding,self).__init__()
        self.dim = dim
        self.num = num
        z_list = []
        for i in range(self.num):
            z_list.append(nn.Parameter(torch.randn(self.dim)))
        self.z_list = nn.ParameterList(z_list).to(device)
    def forward(self):
        pass
class Super_Embedding(Embedding):
    def __init__(self, num, dim, device):
        super(Super_Embedding, self).__init__(num, dim, device)
        pass
    def forward(self):
        pass

'''
300*301*3*(48+16) = 300*301*192
16: n_lamb_sigma
48: n_lamb_sh
300*301 ~~=voxel_init
'''
class BaseFilterGenerator(nn.Module):
    def __init__(self, z_num = 1, z_dim = 64, g_depth = 4,g_width = 256,g_out = 1, n_lamb_sigma = 16, n_lamb_sh = 48, n_compress_size = 16, device = 'cuda:0'):
        super(BaseFilterGenerator,self).__init__()
        self.z_num = z_num
        self.z_dim = z_dim
        self.emb = Embedding(num = self.z_num, dim = self.z_dim, device = 'cuda:0')
        self.filter_generator = self.init_filter_generator(self.z_num, self.z_dim,g_depth,g_width,n_lamb_sigma * n_lamb_sh ,n_compress_size,device)
    def forward(self):
        pass
    def init_filter_generator(self,z_num,z_dim,g_depth,g_width,g_out,n_compress_size, device):
        '''
        for intutive how many latent code z, how many filter for it.
        '''
        filter_generator = nn.ParameterList()
        for i in range(z_num):
            return  nn.ModuleList([nn.Linear(z_dim,g_width)] + \
                                  [nn.Linear(g_width,g_width) if (i not in [g_depth//2]) else nn.Linear(g_width + z_dim, g_width) for i in range(g_depth-2)] + \
                                  [nn.Linear(g_width,g_out)] ).to(self.device)


if __name__ == '__main__':
    g_depth = 8
    g_width = 256
    bfg = BaseFilterGenerator(z_num = 1, z_dim = 64, device = 'cuda:0')
    print(bfg.emb.z_list[0].shape)
