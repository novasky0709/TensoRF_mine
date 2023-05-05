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
    def __init__(self, D = 8,W = 256, device = 'cuda:0',pos_pe = 5, dir_pe = 3,distance_scale =25, rayMarch_weight_thres = 0.0001):
        super(VanillaNeRF,self).__init__()
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.distance_scale = distance_scale
        self.D = D
        self.W = W
        self.device = device
        self.pos_pe = pos_pe
        self.dir_pe = dir_pe
        self.pos_embed_fn, pos_dim_pe = utils.get_embedder(self.pos_pe, 0, input_dims=3)
        self.dir_embed_fn, dir_dim_pe = utils.get_embedder(self.dir_pe, 0, input_dims=3)
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
            validsigma = torch.sigmoid(self.density_linear(x))
            sigma[ray_valid] = validsigma.squeeze(dim=-1)
        alpha, weight, bg_weight = raw2alpha(sigma, dists*self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            input_dir = self.dir_embed_fn(viewdir_sampled[app_mask])
            app_feat[ray_valid] = hidden_feat
            app_feat_valid = torch.cat([app_feat[app_mask],input_dir],dim = -1)
            valid_rgbs = F.relu(self.app_linear(app_feat_valid))
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


if __name__ == '__main__':
    # rays_train [4096,6], rgb_train [4096,3], rgb_maps [4096,3] depth_maps [4096]
    vn = VanillaNeRF()
    rays_train = torch.rand([4096,6] ,device = 'cuda:0')
    rgb_train = torch.rand([4096,1] ,device = 'cuda:0')
    density, app, hidden_feat = vn(rays_train)
    print(density.shape)
    print(app.shape)
    print(hidden_feat.shape)
