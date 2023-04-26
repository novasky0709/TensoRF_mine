import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime



class ResidueBase(torch.nn.Module):
    def __init__(self, src_kwargs,dst_kwargs,**kwargs):
        super(ResidueBase, self).__init__()
        # self.density_n_comp = kwargs['density_n_comp'] # [16,16,16]
        # self.gridSize = kwargs['gridSize'] # [300,300,300]
        # self.appearance_n_comp = kwargs['appearance_n_comp']
        # self.app_dim = kwargs['app_dim']
        # self.fea2denseAct = kwargs['fea2denseAct']
        # self.shadingMode = kwargs['shadingMode']
        # self.pos_pe = kwargs['pos_pe']
        # self.view_pe = kwargs['view_pe']
        # self.featureC = kwargs['featureC']
        # self.device = kwargs['device']

        # self.init_morph_mat()
    def init_morph_mat(self):

        # plane_morph_mat
        pass
    def __len__(self):
        pass
    def __getitem__(self, item):
        pass

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        pass
    def save(self, path):
       pass