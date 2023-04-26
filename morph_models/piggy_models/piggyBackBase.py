import torch
import numpy as np
import torch.nn as nn
from models.tensoRF import *
from morph_models.residue_models.residueTensorVMSplit import *



class PiggyBackBase(torch.nn.Module):
    def __init__(self, src_kwargs,dst_kwargs,tensorf_model = None, train_strategy = 'distill',piggyback_num = 0, device = 'cuda:0',density_D = 8, density_W = 256, app_D = 8, app_W = 256, density_pe = 5, app_pe = 3,**kwargs):
        super(PiggyBackBase, self).__init__()
        self.device = device
        self.train_strategy = train_strategy
        self.tensorf_model = tensorf_model
        residue_module_args = {'src_kwargs':src_kwargs,'dst_kwargs':dst_kwargs,'device':device,'density_D':density_D,'density_W':density_W,'app_D':app_D,'app_W':app_W,'density_pe':density_pe,'app_pe':app_pe}
        self.piggyback_list = []
        if piggyback_num > 0:
            self.init_residue_module(piggyback_num,residue_module_args)

    def raw2alpha(sigma, dist):
        # sigma, dist  [N_rays, N_samples]
        alpha = 1. - torch.exp(-sigma * dist)

        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

        weights = alpha * T[:, :-1]  # [N_rays, N_samples]
        return alpha, weights, T[:, -1:]
    # some initialize code
    def init_residue_module(self):
        pass

    # some inference code
    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        if self.train_strategy == 'distill':
            res = self.distill( rays_chunk, white_bg, is_train, ndc_ray, N_samples)
        elif self.train_strategy == 'finetune':
            res = self.finetune(rays_chunk, white_bg, is_train, ndc_ray, N_samples)
        return res
    def distill(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        '''
        for example:
         input :[N, 6 (pos + view) ] Attention: this is the camera pose, not the point
         ->Tensorf search->[N,k,1+27(or 3)]->residue module[N,k,1+27(or 3)]->[N,k,4]
         so k is the sample point per ray. Though we actually dont need sample point on a ray. We
         try the best to write least code :D
        so what type of result ? [B,N,4(sigma + rgb)]
        '''
        with torch.no_grad():
            _, _, rgb, sigma, _, _, _ = self.tensorf_model(rays_chunk, white_bg, is_train, ndc_ray, N_samples)
        x = torch.cat([sigma[...,None],rgb],dim=-1)
        for pbm in self.piggyback_list:
            x = pbm(x)
        sigma, rgb = torch.split(x, [1, 3], dim=-1)
        alpha, weight, bg_weight = raw2alpha(sigma.squeeze(-1), self.tensorf_model.dists * self.tensorf_model.distance_scale)
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        with torch.no_grad():
            depth_map = torch.sum(weight * self.tensorf_model.z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
        return rgb_map, depth_map , rgb, sigma, alpha, weight, bg_weight
    def finetune(self,rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        '''
        for example:
         input :[N, 6 (pos + view) ]
         ->Tensorf search->[N,1+27(or 3)]->residue module[B,N,1+27(or 3)]->[B,N,4]->render->[N,3]
        '''
        pass
        rgb, _, rgb, sigma, _, _, _ = self.tensorf_model(rays_chunk,white_bg,is_train,ndc_ray,N_samples)

    # def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
    #     grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
    #                      {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
    #     if isinstance(self.renderModule, torch.nn.Module):
    #         grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
    #     return grad_vars
    def get_optparam_groups(self, lr_init_network=0.001):
        grad_vars = []
        for pb_model in self.piggyback_list:
            grad_vars += pb_model.get_optparam_groups(lr_init_network)
        return grad_vars


if __name__ == "__main__":
    pass