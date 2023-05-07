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

class VanillaNeRF(torch.nn.Module):
    def __init__(self, aabb, gridSize, D = 8,W = 256, device = 'cuda:0',pos_pe = 5, dir_pe = 3,distance_scale =25, rayMarch_weight_thres = 0.0001, near_far=[2.0, 6.0], density_shift = -10,step_ratio = 0.5):
        super(VanillaNeRF,self).__init__()
        self.aabb = aabb
        self.density_shift = density_shift
        self.near_far = near_far
        self.device = device
        self.step_ratio = step_ratio
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.distance_scale = distance_scale
        self.D = D
        self.W = W
        self.device = device
        self.pos_pe = pos_pe
        self.dir_pe = dir_pe
        self.pos_embed_fn, pos_dim_pe = utils.get_embedder(self.pos_pe, 0, input_dims=3)
        self.dir_embed_fn, dir_dim_pe = utils.get_embedder(self.dir_pe, 0, input_dims=3)

        self.update_stepSize(gridSize)
        self.init_nn(pos_dim_pe, dir_dim_pe)
        self.set_device(device)
    def forward(self,ray_sampled, xyz_sampled, viewdir_sampled, z_vals, ray_valid,white_bg=True, is_train=False, ndc_ray=False):
        input_pos = self.pos_embed_fn(xyz_sampled[ray_valid])

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        app_feat = torch.zeros((*xyz_sampled.shape[:2], 27), device=xyz_sampled.device)
        sigma_feature = None

        if ray_valid.any():
            x = input_pos
            for i,l in enumerate(self.encoder):
                x = self.encoder[i](x)
                x = F.relu(x)
                if i in [self.D // 2]:
                    x = torch.cat([input_pos,x],dim = -1)
            hidden_feat = x
            validsigma = F.softplus(self.density_linear(x) + self.density_shift)
            sigma[ray_valid] = validsigma.squeeze(dim=-1)
        alpha, weight, bg_weight = raw2alpha(sigma, dists*self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            input_dir = self.dir_embed_fn(viewdir_sampled[app_mask])
            app_feat[ray_valid] = hidden_feat
            app_feat_valid = torch.cat([app_feat[app_mask],input_dir],dim = -1)
            valid_rgbs = torch.sigmoid(self.app_linear(app_feat_valid))
            rgb[app_mask] = valid_rgbs
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * ray_sampled[..., -1]

        return rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight, sigma_feature, app_feat
        # return sigma, app, hidden_feat

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0] # 3 3 3
        self.invaabbSize = 2.0/self.aabbSize # 0.6667
        self.gridSize= gridSize.long().to(self.device) #128 128 128
        self.units=self.aabbSize / (self.gridSize-1) #0.0236
        self.stepSize=torch.mean(self.units)*self.step_ratio #0.0118->这个数错了，导致采样不对了
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize))) # 5.19
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1 #440
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_nn(self,pos_dim_pe, dir_dim_pe):
       self.encoder = nn.ModuleList([nn.Linear(pos_dim_pe,self.W)] + [nn.Linear(self.W,self.W) if (i not in [self.D//2 ]) else nn.Linear(self.W + pos_dim_pe,self.W) for i in range(self.D - 1)] + [nn.Linear(self.W , 27)])
       self.density_linear = nn.Linear(27,1)
       self.app_linear = nn.Linear(27 + dir_dim_pe,3)

    def set_device(self,device):
        for var in dir(self):
            if not var.startswith('_') and isinstance(getattr(self,var), nn.Module):
                getattr(self,var).to(device)

    def get_optparam_groups(self, lr_init_network = 0.001):
        grad_vars = [{'params': self.encoder.parameters(), 'lr': lr_init_network}, {'params': self.density_linear.parameters(), 'lr': lr_init_network},
                         {'params': self.app_linear.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'D': self.D,
            'W': self.W,
            'pos_pe': self.pos_pe,
            'dir_pe':self.dir_pe,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'near_far': self.near_far,
            'density_shift': self.density_shift,
            'step_ratio': self.step_ratio,
        }
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

if __name__ == '__main__':
    # rays_train [4096,6], rgb_train [4096,3], rgb_maps [4096,3] depth_maps [4096]
    vn = VanillaNeRF()
    rays_train = torch.rand([4096,6] ,device = 'cuda:0')
    rgb_train = torch.rand([4096,1] ,device = 'cuda:0')
    density, app, hidden_feat = vn(rays_train)
    print(density.shape)
    print(app.shape)
    print(hidden_feat.shape)
