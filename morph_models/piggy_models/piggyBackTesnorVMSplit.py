import torch
import numpy as np
import torch.nn as nn
from models.tensoRF import *
from models.tensoRF import TensorVMSplit
from morph_models.residue_models.residueTensorVMSplit import ResidueTensorVMSplit
from morph_models.piggy_models.piggyBackBase import *


class PiggyBackTensorVMSplit(PiggyBackBase):
    def __init__(self, src_kwargs,dst_kwargs, tensorf_model = None, train_strategy = 'distill',piggyback_num = 0, device = 'cuda:0',density_D = 8, density_W = 256, app_D = 8, app_W = 256, density_pe = 5, app_pe = 3):
        super(PiggyBackTensorVMSplit, self).__init__(src_kwargs, dst_kwargs,tensorf_model = tensorf_model,train_strategy = train_strategy,piggyback_num = piggyback_num, device = device, density_D = density_D, density_W = density_W, app_D = app_D, app_W = app_W, density_pe = density_pe, app_pe = app_pe)



    def init_residue_module(self,piggyback_num,residue_module_args):
        for i in range(piggyback_num):
            residue_module = ResidueTensorVMSplit(residue_module_args['src_kwargs'],residue_module_args['dst_kwargs'],residue_module_args['device'],residue_module_args['density_D'],\
                                                  residue_module_args['density_W'],residue_module_args['app_D'],residue_module_args['app_W'],residue_module_args['density_pe'],residue_module_args['app_pe']).to(self.device)
            self.piggyback_list.append(residue_module)
        self.piggyback_list = nn.ModuleList(self.piggyback_list)
    # def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
    #     grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
    #                      {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
    #     if isinstance(self.renderModule, torch.nn.Module):
    #         grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
    #     return grad_vars
    # def get_kwargs(self):
    #     return {
    #         'tensorf_model':self.tensorf_model.get_kwargs()
    #     }
    #
    def save(self, path):
        tensorf_kwargs = self.tensorf_model.get_kwargs()
        ckpt = {'tensorf_kwargs': tensorf_kwargs, 'state_dict': self.state_dict()}
        try:
            if self.alphaMask is not None:
                alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
                ckpt.update({'alphaMask.shape': alpha_volume.shape})
                ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
                ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        except:
            pass
        torch.save(ckpt, path)
if __name__ == '__main__':
    src_kwargs = {'app_dim':3}
    dst_kwargs = {'app_dim':3}
    lego_ckpt = torch.load('/home/yuze/Documents/project/TensoRF/log/morph_lego2ficus/tensorf_lego_VM.th', map_location='cuda:0')
    lego_kwargs = lego_ckpt['kwargs']
    lego_kwargs.update({'device':'cuda:0'})
    lego_tensorf = TensorVMSplit(**lego_kwargs)
    lego_tensorf.load(lego_ckpt)
    pbm = PiggyBackTensorVMSplit(src_kwargs,dst_kwargs,tensorf_model=lego_tensorf,train_strategy = 'distill', piggyback_num = 3)
    data = torch.rand([32,6]).to('cuda:0')
    x = pbm(data)
    print(x.shape)