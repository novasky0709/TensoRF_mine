import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

class BaseNeRF(torch.nn.Module):
    def __init__(self,aabb,gridSize, density_shift = -10, near_far = [2.0, 6.0], device = 'cuda:0',step_ratio = 0.5, rayMarch_weight_thres = 0.0001, distance_scale = 25 ):
        super(BaseNeRF,self).__init__()
        self.aabb = aabb
        self.density_shift = density_shift
        self.near_far = near_far
        self.device = device
        self.step_ratio = step_ratio
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.distance_scale = distance_scale
        self.update_stepSize(gridSize)
    def forward(self):
        raise NotImplementedError
    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0] # [3,3,3]
        self.invaabbSize = 2.0/self.aabbSize # [0.667 * 3]
        if not isinstance(gridSize,torch.Tensor) :
            gridSize = torch.Tensor(gridSize)# 300
        self.gridSize= gridSize.long().to(self.device) # 300 300 300
        self.units=self.aabbSize / (self.gridSize-1) #0.01
        self.stepSize=torch.mean(self.units)*self.step_ratio #0.005
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1 # 1036
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)
    def init_nn(self):
        raise NotImplementedError
    def set_device(self,device):
        for var in dir(self):
            if not var.startswith('_') and isinstance(getattr(self,var), nn.Module):
                getattr(self,var).to(device)
    def get_optparam_groups(self, lr_init_network=0.001):
        NotImplementedError
    def get_kwargs(self):
        NotImplementedError
    def save(self,path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)
    def load(self, ckpt):
        self.load_state_dict(ckpt['state_dict'])
    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float() # [1,N_samples]:[0,1,2,3,4,5,6,.....,N_samples-1]
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1) # [2048,1147] : [N_pixel_sample_per_image,N_point_sample_per_ray]
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox
